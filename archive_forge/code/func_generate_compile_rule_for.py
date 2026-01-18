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
def generate_compile_rule_for(self, langname, compiler):
    if langname == 'java':
        self.generate_java_compile_rule(compiler)
        return
    if langname == 'cs':
        if self.environment.machines.matches_build_machine(compiler.for_machine):
            self.generate_cs_compile_rule(compiler)
        return
    if langname == 'vala':
        self.generate_vala_compile_rules(compiler)
        return
    if langname == 'rust':
        self.generate_rust_compile_rules(compiler)
        return
    if langname == 'swift':
        if self.environment.machines.matches_build_machine(compiler.for_machine):
            self.generate_swift_compile_rules(compiler)
        return
    if langname == 'cython':
        self.generate_cython_compile_rules(compiler)
        return
    crstr = self.get_rule_suffix(compiler.for_machine)
    options = self._rsp_options(compiler)
    if langname == 'fortran':
        self.generate_fortran_dep_hack(crstr)
        options['extra'] = 'restat = 1'
    rule = self.compiler_to_rule_name(compiler)
    if langname == 'cuda':
        depargs = NinjaCommandArg.list(compiler.get_dependency_gen_args('$CUDA_ESCAPED_TARGET', '$DEPFILE'), Quoting.none)
    else:
        depargs = NinjaCommandArg.list(compiler.get_dependency_gen_args('$out', '$DEPFILE'), Quoting.none)
    command = compiler.get_exelist()
    args = ['$ARGS'] + depargs + NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none) + compiler.get_compile_only_args() + ['$in']
    description = f'Compiling {compiler.get_display_language()} object $out'
    if compiler.get_argument_syntax() == 'msvc':
        deps = 'msvc'
        depfile = None
    else:
        deps = 'gcc'
        depfile = '$DEPFILE'
    self.add_rule(NinjaRule(rule, command, args, description, **options, deps=deps, depfile=depfile))