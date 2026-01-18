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
def detect_cs_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from . import cs
    compilers, ccache, exe_wrap = _get_compilers(env, 'cs', for_machine)
    popen_exceptions = {}
    info = env.machines[for_machine]
    for comp in compilers:
        try:
            p, out, err = Popen_safe_logged(comp + ['--version'], msg='Detecting compiler via')
        except OSError as e:
            popen_exceptions[join_args(comp + ['--version'])] = e
            continue
        version = search_version(out)
        cls: T.Type[cs.CsCompiler]
        if 'Mono' in out:
            cls = cs.MonoCompiler
        elif 'Visual C#' in out:
            cls = cs.VisualStudioCsCompiler
        else:
            continue
        env.coredata.add_lang_args(cls.language, cls, for_machine, env)
        return cls(comp, version, for_machine, info)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')