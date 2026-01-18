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
def generate_coverage_command(self, elem, outputs: T.List[str], gcovr_exe: T.Optional[str], llvm_cov_exe: T.Optional[str]):
    targets = self.build.get_targets().values()
    use_llvm_cov = False
    exe_args = []
    if gcovr_exe is not None:
        exe_args += ['--gcov', gcovr_exe]
    if llvm_cov_exe is not None:
        exe_args += ['--llvm-cov', llvm_cov_exe]
    for target in targets:
        if not hasattr(target, 'compilers'):
            continue
        for compiler in target.compilers.values():
            if compiler.get_id() == 'clang' and (not compiler.info.is_darwin()):
                use_llvm_cov = True
                break
    elem.add_item('COMMAND', self.environment.get_build_command() + ['--internal', 'coverage'] + outputs + [self.environment.get_source_dir(), os.path.join(self.environment.get_source_dir(), self.build.get_subproject_dir()), self.environment.get_build_dir(), self.environment.get_log_dir()] + exe_args + (['--use-llvm-cov'] if use_llvm_cov else []))