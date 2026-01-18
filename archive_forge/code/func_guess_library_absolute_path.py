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
@lru_cache(maxsize=None)
def guess_library_absolute_path(self, linker, libname, search_dirs, patterns) -> Path:
    from ..compilers.c import CCompiler
    for d in search_dirs:
        for p in patterns:
            trial = CCompiler._get_trials_from_pattern(p, d, libname)
            if not trial:
                continue
            trial = CCompiler._get_file_from_list(self.environment, trial)
            if not trial:
                continue
            return trial