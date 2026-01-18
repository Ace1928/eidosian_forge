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
def generate_rust_compile_rules(self, compiler):
    rule = self.compiler_to_rule_name(compiler)
    command = compiler.get_exelist() + ['$ARGS', '$in']
    description = 'Compiling Rust source $in'
    depfile = '$targetdep'
    depstyle = 'gcc'
    self.add_rule(NinjaRule(rule, command, [], description, deps=depstyle, depfile=depfile))