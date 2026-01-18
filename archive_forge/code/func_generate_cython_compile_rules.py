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
def generate_cython_compile_rules(self, compiler: 'Compiler') -> None:
    rule = self.compiler_to_rule_name(compiler)
    description = 'Compiling Cython source $in'
    command = compiler.get_exelist()
    depargs = compiler.get_dependency_gen_args('$out', '$DEPFILE')
    depfile = '$out.dep' if depargs else None
    args = depargs + ['$ARGS', '$in']
    args += NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none)
    self.add_rule(NinjaRule(rule, command + args, [], description, depfile=depfile, extra='restat = 1'))