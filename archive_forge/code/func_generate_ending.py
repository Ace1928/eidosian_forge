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
def generate_ending(self) -> None:
    for targ, deps in [('all', self.get_build_by_default_targets()), ('meson-test-prereq', self.get_testlike_targets()), ('meson-benchmark-prereq', self.get_testlike_targets(True))]:
        targetlist = []
        if targ == 'all':
            targetlist.extend(['meson-test-prereq', 'meson-benchmark-prereq'])
        for t in deps.values():
            if isinstance(t, build.SharedLibrary) and t.aix_so_archive:
                if self.environment.machines[t.for_machine].is_aix():
                    linker, stdlib_args = self.determine_linker_and_stdlib_args(t)
                    t.get_outputs()[0] = linker.get_archive_name(t.get_outputs()[0])
            targetlist.append(os.path.join(self.get_target_dir(t), t.get_outputs()[0]))
        elem = NinjaBuildElement(self.all_outputs, targ, 'phony', targetlist)
        self.add_build(elem)
    elem = self.create_phony_target('clean', 'CUSTOM_COMMAND', 'PHONY')
    elem.add_item('COMMAND', self.ninja_command + ['-t', 'clean'])
    elem.add_item('description', 'Cleaning')
    ctlist = []
    for t in self.build.get_targets().values():
        if isinstance(t, build.CustomTarget):
            for o in t.get_outputs():
                ctlist.append(os.path.join(self.get_target_dir(t), o))
    if ctlist:
        elem.add_dep(self.generate_custom_target_clean(ctlist))
    if OptionKey('b_coverage') in self.environment.coredata.options and self.environment.coredata.options[OptionKey('b_coverage')].value:
        self.generate_gcov_clean()
        elem.add_dep('clean-gcda')
        elem.add_dep('clean-gcno')
    self.add_build(elem)
    deps = self.get_regen_filelist()
    elem = NinjaBuildElement(self.all_outputs, 'build.ninja', 'REGENERATE_BUILD', deps)
    elem.add_item('pool', 'console')
    self.add_build(elem)
    if self.implicit_meson_outs:
        elem = NinjaBuildElement(self.all_outputs, 'meson-implicit-outs', 'phony', self.implicit_meson_outs)
        self.add_build(elem)
    elem = NinjaBuildElement(self.all_outputs, 'reconfigure', 'REGENERATE_BUILD', 'PHONY')
    elem.add_item('pool', 'console')
    self.add_build(elem)
    elem = NinjaBuildElement(self.all_outputs, deps, 'phony', '')
    self.add_build(elem)