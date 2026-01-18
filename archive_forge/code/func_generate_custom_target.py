from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def generate_custom_target(self, target: build.CustomTarget):
    self.custom_target_generator_inputs(target)
    srcs, ofilenames, cmd = self.eval_custom_target_command(target)
    deps = self.unwrap_dep_list(target)
    deps += self.get_target_depend_files(target)
    if target.build_always_stale:
        deps.append('PHONY')
    if target.depfile is None:
        rulename = 'CUSTOM_COMMAND'
    else:
        rulename = 'CUSTOM_COMMAND_DEP'
    elem = NinjaBuildElement(self.all_outputs, ofilenames, rulename, srcs)
    elem.add_dep(deps)
    for d in target.extra_depends:
        for output in d.get_outputs():
            elem.add_dep(os.path.join(self.get_target_dir(d), output))
    cmd, reason = self.as_meson_exe_cmdline(target.command[0], cmd[1:], extra_bdeps=target.get_transitive_build_target_deps(), capture=ofilenames[0] if target.capture else None, feed=srcs[0] if target.feed else None, env=target.env, verbose=target.console)
    if reason:
        cmd_type = f' (wrapped by meson {reason})'
    else:
        cmd_type = ''
    if target.depfile is not None:
        depfile = target.get_dep_outname(elem.infilenames)
        rel_dfile = os.path.join(self.get_target_dir(target), depfile)
        abs_pdir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
        os.makedirs(abs_pdir, exist_ok=True)
        elem.add_item('DEPFILE', rel_dfile)
    if target.console:
        elem.add_item('pool', 'console')
    full_name = Path(target.subdir, target.name).as_posix()
    elem.add_item('COMMAND', cmd)
    elem.add_item('description', target.description.format(full_name) + cmd_type)
    self.add_build(elem)
    self.processed_targets.add(target.get_id())