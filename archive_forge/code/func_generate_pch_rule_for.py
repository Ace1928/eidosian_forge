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
def generate_pch_rule_for(self, langname, compiler):
    if langname not in {'c', 'cpp'}:
        return
    rule = self.compiler_to_pch_rule_name(compiler)
    depargs = compiler.get_dependency_gen_args('$out', '$DEPFILE')
    if compiler.get_argument_syntax() == 'msvc':
        output = []
    else:
        output = NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none)
    if 'mwcc' in compiler.id:
        output[0].s = '-precompile'
        command = compiler.get_exelist() + ['$ARGS'] + depargs + output + ['$in']
    else:
        command = compiler.get_exelist() + ['$ARGS'] + depargs + output + compiler.get_compile_only_args() + ['$in']
    description = 'Precompiling header $in'
    if compiler.get_argument_syntax() == 'msvc':
        deps = 'msvc'
        depfile = None
    else:
        deps = 'gcc'
        depfile = '$DEPFILE'
    self.add_rule(NinjaRule(rule, command, [], description, deps=deps, depfile=depfile))