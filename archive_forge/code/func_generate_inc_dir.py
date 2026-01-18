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
def generate_inc_dir(self, compiler: 'Compiler', d: str, basedir: str, is_system: bool) -> T.Tuple['ImmutableListProtocol[str]', 'ImmutableListProtocol[str]']:
    if d not in ('', '.'):
        expdir = os.path.normpath(os.path.join(basedir, d))
    else:
        expdir = basedir
    srctreedir = os.path.normpath(os.path.join(self.build_to_src, expdir))
    sargs = compiler.get_include_args(srctreedir, is_system)
    if os.path.isdir(os.path.join(self.environment.get_build_dir(), expdir)):
        bargs = compiler.get_include_args(expdir, is_system)
    else:
        bargs = []
    return (sargs, bargs)