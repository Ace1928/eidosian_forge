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
def detect_cython_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    """Search for a cython compiler."""
    from .cython import CythonCompiler
    compilers, _, _ = _get_compilers(env, 'cython', MachineChoice.BUILD)
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    popen_exceptions: T.Dict[str, Exception] = {}
    for comp in compilers:
        try:
            _, out, err = Popen_safe_logged(comp + ['-V'], msg='Detecting compiler via')
        except OSError as e:
            popen_exceptions[join_args(comp + ['-V'])] = e
            continue
        version: T.Optional[str] = None
        if 'Cython' in out:
            version = search_version(out)
        elif 'Cython' in err:
            version = search_version(err)
        if version is not None:
            comp_class = CythonCompiler
            env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
            return comp_class([], comp, version, for_machine, info, is_cross=is_cross)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')