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
def generate_common_compile_args_per_src_type(self, target: build.BuildTarget) -> dict[str, list[str]]:
    src_type_to_args = {}
    use_pch = self.target_uses_pch(target)
    for src_type_str in target.compilers.keys():
        compiler = target.compilers[src_type_str]
        commands = self._generate_single_compile_base_args(target, compiler)
        if use_pch and 'mw' not in compiler.id:
            commands += self.get_pch_include_args(compiler, target)
        commands += self._generate_single_compile_target_args(target, compiler)
        if use_pch and 'mw' in compiler.id:
            commands += self.get_pch_include_args(compiler, target)
        commands = commands.compiler.compiler_args(commands)
        src_type_to_args[src_type_str] = commands.to_native()
    return src_type_to_args