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
def generate_compile_rules(self):
    for for_machine in MachineChoice:
        clist = self.environment.coredata.compilers[for_machine]
        for langname, compiler in clist.items():
            if compiler.get_id() == 'clang':
                self.generate_llvm_ir_compile_rule(compiler)
            self.generate_compile_rule_for(langname, compiler)
            self.generate_pch_rule_for(langname, compiler)
            for mode in compiler.get_modes():
                self.generate_compile_rule_for(langname, mode)