from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def get_executable_serialisation(self, cmd: T.Sequence[T.Union[programs.ExternalProgram, build.BuildTarget, build.CustomTarget, File, str]], workdir: T.Optional[str]=None, extra_bdeps: T.Optional[T.List[build.BuildTarget]]=None, capture: T.Optional[str]=None, feed: T.Optional[str]=None, env: T.Optional[mesonlib.EnvironmentVariables]=None, tag: T.Optional[str]=None, verbose: bool=False, installdir_map: T.Optional[T.Dict[str, str]]=None) -> 'ExecutableSerialisation':
    exe, *raw_cmd_args = cmd
    if isinstance(exe, programs.ExternalProgram):
        exe_cmd = exe.get_command()
        exe_for_machine = exe.for_machine
    elif isinstance(exe, build.BuildTarget):
        exe_cmd = [self.get_target_filename_abs(exe)]
        exe_for_machine = exe.for_machine
    elif isinstance(exe, build.CustomTarget):
        exe_cmd = [self.get_target_filename_abs(exe)]
        exe_for_machine = MachineChoice.BUILD
    elif isinstance(exe, mesonlib.File):
        exe_cmd = [exe.rel_to_builddir(self.environment.source_dir)]
        exe_for_machine = MachineChoice.BUILD
    else:
        exe_cmd = [exe]
        exe_for_machine = MachineChoice.BUILD
    cmd_args: T.List[str] = []
    for c in raw_cmd_args:
        if isinstance(c, programs.ExternalProgram):
            p = c.get_path()
            assert isinstance(p, str)
            cmd_args.append(p)
        elif isinstance(c, (build.BuildTarget, build.CustomTarget)):
            cmd_args.append(self.get_target_filename_abs(c))
        elif isinstance(c, mesonlib.File):
            cmd_args.append(c.rel_to_builddir(self.environment.source_dir))
        else:
            cmd_args.append(c)
    machine = self.environment.machines[exe_for_machine]
    if machine.is_windows() or machine.is_cygwin():
        extra_paths = self.determine_windows_extra_paths(exe, extra_bdeps or [])
    else:
        extra_paths = []
    is_cross_built = not self.environment.machines.matches_build_machine(exe_for_machine)
    if is_cross_built and self.environment.need_exe_wrapper():
        exe_wrapper = self.environment.get_exe_wrapper()
        if not exe_wrapper or not exe_wrapper.found():
            msg = 'An exe_wrapper is needed but was not found. Please define one in cross file and check the command and/or add it to PATH.'
            raise MesonException(msg)
    else:
        if exe_cmd[0].endswith('.jar'):
            exe_cmd = ['java', '-jar'] + exe_cmd
        elif exe_cmd[0].endswith('.exe') and (not (mesonlib.is_windows() or mesonlib.is_cygwin() or mesonlib.is_wsl())):
            exe_cmd = ['mono'] + exe_cmd
        exe_wrapper = None
    workdir = workdir or self.environment.get_build_dir()
    return ExecutableSerialisation(exe_cmd + cmd_args, env, exe_wrapper, workdir, extra_paths, capture, feed, tag, verbose, installdir_map)