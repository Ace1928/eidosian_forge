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
def detect_rust_compiler(env: 'Environment', for_machine: MachineChoice) -> RustCompiler:
    from . import rust
    from ..linkers import linkers
    popen_exceptions: T.Dict[str, Exception] = {}
    compilers, _, exe_wrap = _get_compilers(env, 'rust', for_machine)
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    cc = detect_c_compiler(env, for_machine)
    is_link_exe = isinstance(cc.linker, linkers.VisualStudioLikeLinkerMixin)
    override = env.lookup_binary_entry(for_machine, 'rust_ld')
    for compiler in compilers:
        arg = ['--version']
        try:
            out = Popen_safe_logged(compiler + arg, msg='Detecting compiler via')[1]
        except OSError as e:
            popen_exceptions[join_args(compiler + arg)] = e
            continue
        version = search_version(out)
        cls: T.Type[RustCompiler] = rust.RustCompiler
        if 'clippy' in out:
            arg = ['--rustc', '--version']
            try:
                out = Popen_safe(compiler + arg)[1]
            except OSError as e:
                popen_exceptions[join_args(compiler + arg)] = e
                continue
            version = search_version(out)
            cls = rust.ClippyRustCompiler
        if 'rustc' in out:
            if any((a.startswith('linker=') for a in compiler)):
                mlog.warning('Please do not put -C linker= in your compiler command, set rust_ld=command in your cross file or use the RUSTC_LD environment variable, otherwise meson will override your selection.')
            compiler = compiler.copy()
            if override is None:
                extra_args: T.Dict[str, T.Union[str, bool]] = {}
                always_args: T.List[str] = []
                if is_link_exe:
                    compiler.extend(cls.use_linker_args(cc.linker.exelist[0], ''))
                    extra_args['direct'] = True
                    extra_args['machine'] = cc.linker.machine
                else:
                    exelist = cc.linker.exelist + cc.linker.get_always_args()
                    if os.path.basename(exelist[0]) in {'ccache', 'sccache'}:
                        del exelist[0]
                    c = exelist.pop(0)
                    compiler.extend(cls.use_linker_args(c, ''))
                    for l in exelist:
                        compiler.extend(['-C', f'link-arg={l}'])
                assert cc.linker is not None, 'for mypy'
                if is_link_exe:
                    linker = type(cc.linker)(for_machine, always_args, exelist=cc.linker.exelist, version=cc.linker.version, **extra_args)
                else:
                    linker = type(cc.linker)(compiler, for_machine, cc.LINKER_PREFIX, always_args=always_args, version=cc.linker.version, **extra_args)
            elif 'link' in override[0]:
                linker = guess_win_linker(env, override, cls, version, for_machine, use_linker_prefix=False)
                assert isinstance(linker, linkers.VisualStudioLikeLinkerMixin)
                linker.direct = True
                compiler.extend(cls.use_linker_args(linker.exelist[0], ''))
            else:
                cc = _detect_c_or_cpp_compiler(env, 'c', for_machine, override_compiler=override)
                linker = cc.linker
                c = linker.exelist[1] if linker.exelist[0].endswith('ccache') else linker.exelist[0]
                compiler.extend(cls.use_linker_args(c, ''))
            env.coredata.add_lang_args(cls.language, cls, for_machine, env)
            return cls(compiler, version, for_machine, is_cross, info, exe_wrap, linker=linker)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')