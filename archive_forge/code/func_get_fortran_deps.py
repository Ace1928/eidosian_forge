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
def get_fortran_deps(self, compiler: FortranCompiler, src: Path, target) -> T.List[str]:
    """
        Find all module and submodule needed by a Fortran target
        """
    if self.use_dyndeps_for_fortran():
        return []
    dirname = Path(self.get_target_private_dir(target))
    tdeps = self.fortran_deps[target.get_basename()]
    srcdir = Path(self.source_dir)
    mod_files = _scan_fortran_file_deps(src, srcdir, dirname, tdeps, compiler)
    return mod_files