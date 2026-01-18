from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def detect_fortran_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from . import fortran
    from ..linkers import linkers
    popen_exceptions: T.Dict[str, T.Union[Exception, str]] = {}
    compilers, ccache, exe_wrap = _get_compilers(env, 'fortran', for_machine)
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    cls: T.Type[FortranCompiler]
    for compiler in compilers:
        for arg in ['--version', '-V']:
            try:
                p, out, err = Popen_safe_logged(compiler + [arg], msg='Detecting compiler via')
            except OSError as e:
                popen_exceptions[join_args(compiler + [arg])] = e
                continue
            version = search_version(out)
            full_version = out.split('\n', 1)[0]
            guess_gcc_or_lcc: T.Optional[str] = None
            if 'GNU Fortran' in out:
                guess_gcc_or_lcc = 'gcc'
            if 'e2k' in out and 'lcc' in out:
                guess_gcc_or_lcc = 'lcc'
            if guess_gcc_or_lcc:
                defines = _get_gnu_compiler_defines(compiler)
                if not defines:
                    popen_exceptions[join_args(compiler)] = 'no pre-processor defines'
                    continue
                if guess_gcc_or_lcc == 'lcc':
                    version = _get_lcc_version_from_defines(defines)
                    cls = fortran.ElbrusFortranCompiler
                    linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                    return cls(compiler, version, for_machine, is_cross, info, exe_wrap, defines, full_version=full_version, linker=linker)
                else:
                    version = _get_gnu_version_from_defines(defines)
                    cls = fortran.GnuFortranCompiler
                    linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                    return cls(compiler, version, for_machine, is_cross, info, exe_wrap, defines, full_version=full_version, linker=linker)
            if 'Arm C/C++/Fortran Compiler' in out:
                cls = fortran.ArmLtdFlangFortranCompiler
                arm_ver_match = re.search('version (\\d+)\\.(\\d+)\\.?(\\d+)? \\(build number (\\d+)\\)', out)
                assert arm_ver_match is not None, 'for mypy'
                version = '.'.join([x for x in arm_ver_match.groups() if x is not None])
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker)
            if 'G95' in out:
                cls = fortran.G95FortranCompiler
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'Sun Fortran' in err:
                version = search_version(err)
                cls = fortran.SunFortranCompiler
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'Intel(R) Fortran Compiler for applications' in err:
                version = search_version(err)
                target = 'x86' if 'IA-32' in err else 'x86_64'
                cls = fortran.IntelLLVMClFortranCompiler
                env.coredata.add_lang_args(cls.language, cls, for_machine, env)
                linker = linkers.XilinkDynamicLinker(for_machine, [], version=version)
                return cls(compiler, version, for_machine, is_cross, info, target, exe_wrap, linker=linker)
            if 'Intel(R) Visual Fortran' in err or 'Intel(R) Fortran' in err:
                version = search_version(err)
                target = 'x86' if 'IA-32' in err else 'x86_64'
                cls = fortran.IntelClFortranCompiler
                env.coredata.add_lang_args(cls.language, cls, for_machine, env)
                linker = linkers.XilinkDynamicLinker(for_machine, [], version=version)
                return cls(compiler, version, for_machine, is_cross, info, target, exe_wrap, linker=linker)
            if 'ifort (IFORT)' in out:
                cls = fortran.IntelFortranCompiler
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'ifx (IFORT)' in out or 'ifx (IFX)' in out:
                cls = fortran.IntelLLVMFortranCompiler
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'PathScale EKOPath(tm)' in err:
                return fortran.PathScaleFortranCompiler(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version)
            if 'PGI Compilers' in out:
                cls = fortran.PGIFortranCompiler
                env.coredata.add_lang_args(cls.language, cls, for_machine, env)
                linker = linkers.PGIDynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=version)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'NVIDIA Compilers and Tools' in out:
                cls = fortran.NvidiaHPC_FortranCompiler
                env.coredata.add_lang_args(cls.language, cls, for_machine, env)
                linker = linkers.PGIDynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=version)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'flang' in out or 'clang' in out:
                cls = fortran.FlangFortranCompiler
                linker = None
                if 'windows' in out or env.machines[for_machine].is_windows():
                    try:
                        linker = guess_win_linker(env, compiler, cls, version, for_machine, invoked_directly=False)
                    except MesonException:
                        pass
                if linker is None:
                    linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'Open64 Compiler Suite' in err:
                cls = fortran.Open64FortranCompiler
                linker = guess_nix_linker(env, compiler, cls, version, for_machine)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
            if 'NAG Fortran' in err:
                full_version = err.split('\n', 1)[0]
                version = full_version.split()[-1]
                cls = fortran.NAGFortranCompiler
                env.coredata.add_lang_args(cls.language, cls, for_machine, env)
                linker = linkers.NAGDynamicLinker(compiler, for_machine, cls.LINKER_PREFIX, [], version=version)
                return cls(compiler, version, for_machine, is_cross, info, exe_wrap, full_version=full_version, linker=linker)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')