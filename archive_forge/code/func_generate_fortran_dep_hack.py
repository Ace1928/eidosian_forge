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
def generate_fortran_dep_hack(self, crstr: str) -> None:
    if self.use_dyndeps_for_fortran():
        return
    rule = f'FORTRAN_DEP_HACK{crstr}'
    if mesonlib.is_windows():
        cmd = ['cmd', '/C']
    else:
        cmd = ['true']
    self.add_rule_comment(NinjaComment('Workaround for these issues:\nhttps://groups.google.com/forum/#!topic/ninja-build/j-2RfBIOd_8\nhttps://gcc.gnu.org/bugzilla/show_bug.cgi?id=47485'))
    self.add_rule(NinjaRule(rule, cmd, [], 'Dep hack', extra='restat = 1'))