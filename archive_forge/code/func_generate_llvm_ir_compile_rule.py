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
def generate_llvm_ir_compile_rule(self, compiler):
    if self.created_llvm_ir_rule[compiler.for_machine]:
        return
    rule = self.get_compiler_rule_name('llvm_ir', compiler.for_machine)
    command = compiler.get_exelist()
    args = ['$ARGS'] + NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none) + compiler.get_compile_only_args() + ['$in']
    description = 'Compiling LLVM IR object $in'
    options = self._rsp_options(compiler)
    self.add_rule(NinjaRule(rule, command, args, description, **options))
    self.created_llvm_ir_rule[compiler.for_machine] = True