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
def generate_cs_compile_rule(self, compiler: 'CsCompiler') -> None:
    rule = self.compiler_to_rule_name(compiler)
    command = compiler.get_exelist()
    args = ['$ARGS', '$in']
    description = 'Compiling C Sharp target $out'
    self.add_rule(NinjaRule(rule, command, args, description, rspable=mesonlib.is_windows(), rspfile_quote_style=compiler.rsp_file_syntax()))