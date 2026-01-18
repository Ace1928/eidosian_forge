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
def create_test_serialisation(self, tests: T.List['Test']) -> T.List[TestSerialisation]:
    arr: T.List[TestSerialisation] = []
    for t in sorted(tests, key=lambda tst: -1 * tst.priority):
        exe = t.get_exe()
        if isinstance(exe, programs.ExternalProgram):
            cmd = exe.get_command()
        else:
            cmd = [os.path.join(self.environment.get_build_dir(), self.get_target_filename(exe))]
        if isinstance(exe, (build.BuildTarget, programs.ExternalProgram)):
            test_for_machine = exe.for_machine
        else:
            test_for_machine = MachineChoice.BUILD
        for a in t.cmd_args:
            if isinstance(a, build.BuildTarget):
                if a.for_machine is MachineChoice.HOST:
                    test_for_machine = MachineChoice.HOST
                    break
        is_cross = self.environment.is_cross_build(test_for_machine)
        exe_wrapper = self.environment.get_exe_wrapper()
        machine = self.environment.machines[exe.for_machine]
        if machine.is_windows() or machine.is_cygwin():
            extra_bdeps: T.List[T.Union[build.BuildTarget, build.CustomTarget]] = []
            if isinstance(exe, build.CustomTarget):
                extra_bdeps = list(exe.get_transitive_build_target_deps())
            extra_paths = self.determine_windows_extra_paths(exe, extra_bdeps)
            for a in t.cmd_args:
                if isinstance(a, build.BuildTarget):
                    for p in self.determine_windows_extra_paths(a, []):
                        if p not in extra_paths:
                            extra_paths.append(p)
        else:
            extra_paths = []
        cmd_args: T.List[str] = []
        depends: T.Set[build.Target] = set(t.depends)
        if isinstance(exe, build.Target):
            depends.add(exe)
        for a in t.cmd_args:
            if isinstance(a, build.Target):
                depends.add(a)
            elif isinstance(a, build.CustomTargetIndex):
                depends.add(a.target)
            if isinstance(a, mesonlib.File):
                a = os.path.join(self.environment.get_build_dir(), a.rel_to_builddir(self.build_to_src))
                cmd_args.append(a)
            elif isinstance(a, str):
                cmd_args.append(a)
            elif isinstance(a, (build.Target, build.CustomTargetIndex)):
                cmd_args.extend(self.construct_target_rel_paths(a, t.workdir))
            else:
                raise MesonException('Bad object in test command.')
        t_env = copy.deepcopy(t.env)
        if not machine.is_windows() and (not machine.is_cygwin()) and (not machine.is_darwin()):
            ld_lib_path: T.Set[str] = set()
            for d in depends:
                if isinstance(d, build.BuildTarget):
                    for l in d.get_all_link_deps():
                        if isinstance(l, build.SharedLibrary):
                            ld_lib_path.add(os.path.join(self.environment.get_build_dir(), l.get_subdir()))
            if ld_lib_path:
                t_env.prepend('LD_LIBRARY_PATH', list(ld_lib_path), ':')
        ts = TestSerialisation(t.get_name(), t.project_name, t.suite, cmd, is_cross, exe_wrapper, self.environment.need_exe_wrapper(), t.is_parallel, cmd_args, t_env, t.should_fail, t.timeout, t.workdir, extra_paths, t.protocol, t.priority, isinstance(exe, (build.Target, build.CustomTargetIndex)), isinstance(exe, build.Executable), [x.get_id() for x in depends], self.environment.coredata.version, t.verbose)
        arr.append(ts)
    return arr