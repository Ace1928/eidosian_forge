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
def generate_cs_target(self, target: build.BuildTarget):
    fname = target.get_filename()
    outname_rel = os.path.join(self.get_target_dir(target), fname)
    src_list = target.get_sources()
    compiler = target.compilers['cs']
    rel_srcs = [os.path.normpath(s.rel_to_builddir(self.build_to_src)) for s in src_list]
    deps = []
    commands = compiler.compiler_args(target.extra_args['cs'])
    commands += compiler.get_optimization_args(target.get_option(OptionKey('optimization')))
    commands += compiler.get_debug_args(target.get_option(OptionKey('debug')))
    if isinstance(target, build.Executable):
        commands.append('-target:exe')
    elif isinstance(target, build.SharedLibrary):
        commands.append('-target:library')
    else:
        raise MesonException('Unknown C# target type.')
    resource_args, resource_deps = self.generate_cs_resource_tasks(target)
    commands += resource_args
    deps += resource_deps
    commands += compiler.get_output_args(outname_rel)
    for l in target.link_targets:
        lname = os.path.join(self.get_target_dir(l), l.get_filename())
        commands += compiler.get_link_args(lname)
        deps.append(lname)
    if '-g' in commands:
        outputs = [outname_rel, outname_rel + '.mdb']
    else:
        outputs = [outname_rel]
    generated_sources = self.get_target_generated_sources(target)
    generated_rel_srcs = []
    for rel_src in generated_sources.keys():
        if rel_src.lower().endswith('.cs'):
            generated_rel_srcs.append(os.path.normpath(rel_src))
        deps.append(os.path.normpath(rel_src))
    for dep in target.get_external_deps():
        commands.extend_direct(dep.get_link_args())
    commands += self.build.get_project_args(compiler, target.subproject, target.for_machine)
    commands += self.build.get_global_args(compiler, target.for_machine)
    elem = NinjaBuildElement(self.all_outputs, outputs, self.compiler_to_rule_name(compiler), rel_srcs + generated_rel_srcs)
    elem.add_dep(deps)
    elem.add_item('ARGS', commands)
    self.add_build(elem)
    self.generate_generator_list_rules(target)
    self.create_target_source_introspection(target, compiler, commands, rel_srcs, generated_rel_srcs)