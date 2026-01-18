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
def generate_cython_transpile(self, target: build.BuildTarget) -> T.Tuple[T.MutableMapping[str, File], T.MutableMapping[str, File], T.List[str]]:
    """Generate rules for transpiling Cython files to C or C++"""
    static_sources: T.MutableMapping[str, File] = OrderedDict()
    generated_sources: T.MutableMapping[str, File] = OrderedDict()
    cython_sources: T.List[str] = []
    cython = target.compilers['cython']
    args: T.List[str] = []
    args += cython.get_always_args()
    args += cython.get_debug_args(target.get_option(OptionKey('debug')))
    args += cython.get_optimization_args(target.get_option(OptionKey('optimization')))
    args += cython.get_option_compile_args(target.get_options())
    args += self.build.get_global_args(cython, target.for_machine)
    args += self.build.get_project_args(cython, target.subproject, target.for_machine)
    args += target.get_extra_args('cython')
    ext = target.get_option(OptionKey('language', machine=target.for_machine, lang='cython'))
    pyx_sources = []
    for src in target.get_sources():
        if src.endswith('.pyx'):
            output = os.path.join(self.get_target_private_dir(target), f'{src}.{ext}')
            element = NinjaBuildElement(self.all_outputs, [output], self.compiler_to_rule_name(cython), [src.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())])
            element.add_item('ARGS', args)
            self.add_build(element)
            cython_sources.append(output)
            pyx_sources.append(element)
        else:
            static_sources[src.rel_to_builddir(self.build_to_src)] = src
    header_deps = []
    for gen in target.get_generated_sources():
        for ssrc in gen.get_outputs():
            if isinstance(gen, GeneratedList):
                ssrc = os.path.join(self.get_target_private_dir(target), ssrc)
            else:
                ssrc = os.path.join(gen.get_subdir(), ssrc)
            if ssrc.endswith('.pyx'):
                output = os.path.join(self.get_target_private_dir(target), f'{ssrc}.{ext}')
                element = NinjaBuildElement(self.all_outputs, [output], self.compiler_to_rule_name(cython), [ssrc])
                element.add_item('ARGS', args)
                self.add_build(element)
                pyx_sources.append(element)
                cython_sources.append(output)
            else:
                generated_sources[ssrc] = mesonlib.File.from_built_file(gen.get_subdir(), ssrc)
                if not self.environment.is_source(ssrc) and (not self.environment.is_object(ssrc)) and (not self.environment.is_library(ssrc)) and (not modules.is_module_library(ssrc)):
                    header_deps.append(ssrc)
    for source in pyx_sources:
        source.add_orderdep(header_deps)
    return (static_sources, generated_sources, cython_sources)