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
def generate_jar_target(self, target: build.Jar):
    fname = target.get_filename()
    outname_rel = os.path.join(self.get_target_dir(target), fname)
    src_list = target.get_sources()
    resources = target.get_java_resources()
    class_list = []
    compiler = target.compilers['java']
    c = 'c'
    m = 'm'
    e = ''
    f = 'f'
    main_class = target.get_main_class()
    if main_class != '':
        e = 'e'
    generated_sources = self.get_target_generated_sources(target)
    gen_src_list = []
    for rel_src in generated_sources.keys():
        raw_src = File.from_built_relative(rel_src)
        if rel_src.endswith('.java'):
            gen_src_list.append(raw_src)
    compile_args = self.determine_single_java_compile_args(target, compiler)
    for src in src_list + gen_src_list:
        plain_class_path = self.generate_single_java_compile(src, target, compiler, compile_args)
        class_list.append(plain_class_path)
    class_dep_list = [os.path.join(self.get_target_private_dir(target), i) for i in class_list]
    manifest_path = os.path.join(self.get_target_private_dir(target), 'META-INF', 'MANIFEST.MF')
    manifest_fullpath = os.path.join(self.environment.get_build_dir(), manifest_path)
    os.makedirs(os.path.dirname(manifest_fullpath), exist_ok=True)
    with open(manifest_fullpath, 'w', encoding='utf-8') as manifest:
        if any(target.link_targets):
            manifest.write('Class-Path: ')
            cp_paths = [os.path.join(self.get_target_dir(l), l.get_filename()) for l in target.link_targets]
            manifest.write(' '.join(cp_paths))
        manifest.write('\n')
    jar_rule = 'java_LINKER'
    commands = [c + m + e + f]
    commands.append(manifest_path)
    if e != '':
        commands.append(main_class)
    commands.append(self.get_target_filename(target))
    commands += ['-C', self.get_target_private_dir(target), '.']
    elem = NinjaBuildElement(self.all_outputs, outname_rel, jar_rule, [])
    elem.add_dep(class_dep_list)
    if resources:
        elem.add_orderdep(self.__generate_sources_structure(Path(self.get_target_private_dir(target)), resources)[0])
    elem.add_item('ARGS', commands)
    self.add_build(elem)
    self.create_target_source_introspection(target, compiler, compile_args, src_list, gen_src_list)