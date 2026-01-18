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
def generate_run_target(self, target: build.RunTarget):
    target_name = self.build_run_target_name(target)
    if not target.command:
        elem = NinjaBuildElement(self.all_outputs, target_name, 'phony', [])
    else:
        target_env = self.get_run_target_env(target)
        _, _, cmd = self.eval_custom_target_command(target)
        meson_exe_cmd, reason = self.as_meson_exe_cmdline(target.command[0], cmd[1:], env=target_env, verbose=True)
        cmd_type = f' (wrapped by meson {reason})' if reason else ''
        elem = self.create_phony_target(target_name, 'CUSTOM_COMMAND', [])
        elem.add_item('COMMAND', meson_exe_cmd)
        elem.add_item('description', f'Running external command {target.name}{cmd_type}')
        elem.add_item('pool', 'console')
    deps = self.unwrap_dep_list(target)
    deps += self.get_target_depend_files(target)
    elem.add_dep(deps)
    self.add_build(elem)
    self.processed_targets.add(target.get_id())