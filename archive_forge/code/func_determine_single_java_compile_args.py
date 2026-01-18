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
def determine_single_java_compile_args(self, target, compiler):
    args = []
    args += self.build.get_global_args(compiler, target.for_machine)
    args += self.build.get_project_args(compiler, target.subproject, target.for_machine)
    args += target.get_java_args()
    args += compiler.get_output_args(self.get_target_private_dir(target))
    args += target.get_classpath_args()
    curdir = target.get_subdir()
    sourcepath = os.path.join(self.build_to_src, curdir) + os.pathsep
    sourcepath += os.path.normpath(curdir) + os.pathsep
    for i in target.include_dirs:
        for idir in i.get_incdirs():
            sourcepath += os.path.join(self.build_to_src, i.curdir, idir) + os.pathsep
    args += ['-sourcepath', sourcepath]
    return args