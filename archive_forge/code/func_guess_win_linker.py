from __future__ import annotations
from .. import mlog
from ..mesonlib import (
import re
import shlex
import typing as T
def guess_win_linker(env: 'Environment', compiler: T.List[str], comp_class: T.Type['Compiler'], comp_version: str, for_machine: MachineChoice, *, use_linker_prefix: bool=True, invoked_directly: bool=True, extra_args: T.Optional[T.List[str]]=None) -> 'DynamicLinker':
    from . import linkers
    env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
    if not use_linker_prefix or comp_class.LINKER_PREFIX is None:
        check_args = ['/logo', '--version']
    elif isinstance(comp_class.LINKER_PREFIX, str):
        check_args = [comp_class.LINKER_PREFIX + '/logo', comp_class.LINKER_PREFIX + '--version']
    elif isinstance(comp_class.LINKER_PREFIX, list):
        check_args = comp_class.LINKER_PREFIX + ['/logo'] + comp_class.LINKER_PREFIX + ['--version']
    check_args += env.coredata.get_external_link_args(for_machine, comp_class.language)
    override: T.List[str] = []
    value = env.lookup_binary_entry(for_machine, comp_class.language + '_ld')
    if value is not None:
        override = comp_class.use_linker_args(value[0], comp_version)
        check_args += override
    if extra_args is not None:
        check_args.extend(extra_args)
    p, o, _ = Popen_safe(compiler + check_args)
    if 'LLD' in o.split('\n', maxsplit=1)[0]:
        if '(compatible with GNU linkers)' in o:
            return linkers.LLVMDynamicLinker(compiler, for_machine, comp_class.LINKER_PREFIX, override, version=search_version(o))
        elif not invoked_directly:
            return linkers.ClangClDynamicLinker(for_machine, override, exelist=compiler, prefix=comp_class.LINKER_PREFIX, version=search_version(o), direct=False, machine=None)
    if value is not None and invoked_directly:
        compiler = value
    p, o, e = Popen_safe(compiler + check_args)
    if 'LLD' in o.split('\n', maxsplit=1)[0]:
        return linkers.ClangClDynamicLinker(for_machine, [], prefix=comp_class.LINKER_PREFIX if use_linker_prefix else [], exelist=compiler, version=search_version(o), direct=invoked_directly)
    elif 'OPTLINK' in o:
        return linkers.OptlinkDynamicLinker(compiler, for_machine, version=search_version(o))
    elif o.startswith('Microsoft') or e.startswith('Microsoft'):
        out = o or e
        match = re.search('.*(X86|X64|ARM|ARM64).*', out)
        if match:
            target = str(match.group(1))
        else:
            target = 'x86'
        return linkers.MSVCDynamicLinker(for_machine, [], machine=target, exelist=compiler, prefix=comp_class.LINKER_PREFIX if use_linker_prefix else [], version=search_version(out), direct=invoked_directly)
    elif 'GNU coreutils' in o:
        import shutil
        fullpath = shutil.which(compiler[0])
        raise EnvironmentException(f'Found GNU link.exe instead of MSVC link.exe in {fullpath}.\nThis link.exe is not a linker.\nYou may need to reorder entries to your %PATH% variable to resolve this.')
    __failed_to_detect_linker(compiler, check_args, o, e)