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
def generate_msvc_pch_command(self, target, compiler, pch):
    header = pch[0]
    pchname = compiler.get_pch_name(header)
    dst = os.path.join(self.get_target_private_dir(target), pchname)
    commands = []
    commands += self.generate_basic_compiler_args(target, compiler)
    if len(pch) == 1:
        source = self.create_msvc_pch_implementation(target, compiler.get_language(), pch[0])
        pch_header_dir = os.path.dirname(os.path.join(self.build_to_src, target.get_source_subdir(), header))
        commands += compiler.get_include_args(pch_header_dir, False)
    else:
        source = os.path.join(self.build_to_src, target.get_source_subdir(), pch[1])
    just_name = os.path.basename(header)
    objname, pch_args = compiler.gen_pch_args(just_name, source, dst)
    commands += pch_args
    commands += self._generate_single_compile(target, compiler)
    commands += self.get_compile_debugfile_args(compiler, target, objname)
    dep = dst + '.' + compiler.get_depfile_suffix()
    link_objects = [objname] if compiler.should_link_pch_object() else []
    return (commands, dep, dst, link_objects, source)