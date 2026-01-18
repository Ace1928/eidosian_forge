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
def detect_masm_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    is_cross = env.is_cross_build(for_machine)
    cc = detect_c_compiler(env, for_machine)
    if not is_cross:
        from ..environment import detect_machine_info
        info = detect_machine_info({'c': cc})
    else:
        info = env.machines[for_machine]
    from .asm import MasmCompiler, MasmARMCompiler
    comp_class: T.Type[Compiler]
    if info.cpu_family == 'x86':
        comp = ['ml']
        comp_class = MasmCompiler
        arg = '/?'
    elif info.cpu_family == 'x86_64':
        comp = ['ml64']
        comp_class = MasmCompiler
        arg = '/?'
    elif info.cpu_family == 'arm':
        comp = ['armasm']
        comp_class = MasmARMCompiler
        arg = '-h'
    elif info.cpu_family == 'aarch64':
        comp = ['armasm64']
        comp_class = MasmARMCompiler
        arg = '-h'
    else:
        raise EnvironmentException(f'Platform {info.cpu_family} not supported by MASM')
    popen_exceptions: T.Dict[str, Exception] = {}
    try:
        output = Popen_safe(comp + [arg])[2]
        version = search_version(output)
        env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
        return comp_class([], comp, version, for_machine, info, cc.linker, is_cross=is_cross)
    except OSError as e:
        popen_exceptions[' '.join(comp + [arg])] = e
    _handle_exceptions(popen_exceptions, [comp])
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')