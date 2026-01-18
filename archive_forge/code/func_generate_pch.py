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
def generate_pch(self, target, header_deps=None):
    header_deps = header_deps if header_deps is not None else []
    pch_objects = []
    for lang in ['c', 'cpp']:
        pch = target.get_pch(lang)
        if not pch:
            continue
        if not has_path_sep(pch[0]) or not has_path_sep(pch[-1]):
            msg = f'Precompiled header of {target.get_basename()!r} must not be in the same directory as source, please put it in a subdirectory.'
            raise InvalidArguments(msg)
        compiler: Compiler = target.compilers[lang]
        if compiler.get_argument_syntax() == 'msvc':
            commands, dep, dst, objs, src = self.generate_msvc_pch_command(target, compiler, pch)
            extradep = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
        elif compiler.id == 'intel':
            continue
        elif 'mwcc' in compiler.id:
            src = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
            commands, dep, dst, objs = self.generate_mwcc_pch_command(target, compiler, pch[0])
            extradep = None
        else:
            src = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
            commands, dep, dst, objs = self.generate_gcc_pch_command(target, compiler, pch[0])
            extradep = None
        pch_objects += objs
        rulename = self.compiler_to_pch_rule_name(compiler)
        elem = NinjaBuildElement(self.all_outputs, objs + [dst], rulename, src)
        if extradep is not None:
            elem.add_dep(extradep)
        self.add_header_deps(target, elem, header_deps)
        elem.add_item('ARGS', commands)
        elem.add_item('DEPFILE', dep)
        self.add_build(elem)
    return pch_objects