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
def generate_target(self, target):
    try:
        if isinstance(target, build.BuildTarget):
            os.makedirs(self.get_target_private_dir_abs(target))
    except FileExistsError:
        pass
    if isinstance(target, build.CustomTarget):
        self.generate_custom_target(target)
    if isinstance(target, build.RunTarget):
        self.generate_run_target(target)
    compiled_sources = []
    source2object = {}
    name = target.get_id()
    if name in self.processed_targets:
        return
    self.processed_targets.add(name)
    self.introspection_data[name] = {}
    self.process_target_dependencies(target)
    self.generate_shlib_aliases(target, self.get_target_dir(target))
    if isinstance(target, build.Jar):
        self.generate_jar_target(target)
        return
    if target.uses_rust():
        self.generate_rust_target(target)
        return
    if 'cs' in target.compilers:
        self.generate_cs_target(target)
        return
    if 'swift' in target.compilers:
        self.generate_swift_target(target)
        return
    is_compile_target = isinstance(target, build.CompileTarget)
    target_sources: T.MutableMapping[str, File]
    generated_sources: T.MutableMapping[str, File]
    transpiled_sources: T.List[str]
    if 'vala' in target.compilers:
        target_sources, generated_sources, transpiled_sources = self.generate_vala_compile(target)
    elif 'cython' in target.compilers:
        target_sources, generated_sources, transpiled_sources = self.generate_cython_transpile(target)
    else:
        target_sources = self.get_target_sources(target)
        generated_sources = self.get_target_generated_sources(target)
        transpiled_sources = []
    self.scan_fortran_module_outputs(target)
    self.generate_generator_list_rules(target)
    outname = self.get_target_filename(target)
    obj_list = []
    is_unity = target.is_unity
    header_deps = []
    unity_src = []
    unity_deps = []
    header_deps += self.get_generated_headers(target)
    if is_unity:
        langs = set(target.compilers.keys())
        langs_cant = langs.intersection(backends.LANGS_CANT_UNITY)
        if langs_cant:
            langs_are = langs = ', '.join(langs_cant).upper()
            langs_are += ' are' if len(langs_cant) > 1 else ' is'
            msg = f'{langs_are} not supported in Unity builds yet, so {langs} sources in the {target.name!r} target will be compiled normally'
            mlog.log(mlog.red('FIXME'), msg)
    generated_source_files = []
    for rel_src in generated_sources.keys():
        raw_src = File.from_built_relative(rel_src)
        if self.environment.is_source(rel_src):
            if is_unity and self.get_target_source_can_unity(target, rel_src):
                unity_deps.append(raw_src)
                abs_src = os.path.join(self.environment.get_build_dir(), rel_src)
                unity_src.append(abs_src)
            else:
                generated_source_files.append(raw_src)
        elif self.environment.is_object(rel_src):
            obj_list.append(rel_src)
        elif self.environment.is_library(rel_src) or modules.is_module_library(rel_src):
            pass
        elif is_compile_target:
            generated_source_files.append(raw_src)
        else:
            header_deps.append(raw_src)
    d_generated_deps = []
    for src in generated_source_files:
        if self.environment.is_llvm_ir(src):
            o, s = self.generate_llvm_ir_compile(target, src)
        else:
            o, s = self.generate_single_compile(target, src, True, order_deps=header_deps)
        compiled_sources.append(s)
        source2object[s] = o
        obj_list.append(o)
        if s.split('.')[-1] in compilers.lang_suffixes['d']:
            d_generated_deps.append(o)
    use_pch = self.target_uses_pch(target)
    if use_pch and target.has_pch():
        pch_objects = self.generate_pch(target, header_deps=header_deps)
    else:
        pch_objects = []
    o, od = self.flatten_object_list(target)
    obj_targets = [t for t in od if t.uses_fortran()]
    obj_list.extend(o)
    fortran_order_deps = [File(True, *os.path.split(self.get_target_filename(t))) for t in obj_targets]
    fortran_inc_args: T.List[str] = []
    if target.uses_fortran():
        fortran_inc_args = mesonlib.listify([target.compilers['fortran'].get_include_args(self.get_target_private_dir(t), is_system=False) for t in obj_targets])
    transpiled_source_files = []
    for src in transpiled_sources:
        raw_src = File.from_built_relative(src)
        if self.environment.is_header(src):
            header_deps.append(raw_src)
        else:
            transpiled_source_files.append(raw_src)
    for src in transpiled_source_files:
        o, s = self.generate_single_compile(target, src, True, [], header_deps)
        obj_list.append(o)
    for src in target_sources.values():
        if not self.environment.is_header(src) or is_compile_target:
            if self.environment.is_llvm_ir(src):
                o, s = self.generate_llvm_ir_compile(target, src)
                obj_list.append(o)
            elif is_unity and self.get_target_source_can_unity(target, src):
                abs_src = os.path.join(self.environment.get_build_dir(), src.rel_to_builddir(self.build_to_src))
                unity_src.append(abs_src)
            else:
                o, s = self.generate_single_compile(target, src, False, [], header_deps + d_generated_deps + fortran_order_deps, fortran_inc_args)
                obj_list.append(o)
                compiled_sources.append(s)
                source2object[s] = o
    if is_unity:
        for src in self.generate_unity_files(target, unity_src):
            o, s = self.generate_single_compile(target, src, True, unity_deps + header_deps + d_generated_deps, fortran_order_deps, fortran_inc_args, unity_src)
            obj_list.append(o)
            compiled_sources.append(s)
            source2object[s] = o
    if is_compile_target:
        return
    linker, stdlib_args = self.determine_linker_and_stdlib_args(target)
    if isinstance(target, build.StaticLibrary) and target.prelink:
        final_obj_list = self.generate_prelink(target, obj_list)
    else:
        final_obj_list = obj_list
    elem = self.generate_link(target, outname, final_obj_list, linker, pch_objects, stdlib_args=stdlib_args)
    self.generate_dependency_scan_target(target, compiled_sources, source2object, generated_source_files, fortran_order_deps)
    self.add_build(elem)
    if isinstance(target, build.SharedLibrary) and self.environment.machines[target.for_machine].is_aix():
        if target.aix_so_archive:
            elem = NinjaBuildElement(self.all_outputs, linker.get_archive_name(outname), 'AIX_LINKER', [outname])
            self.add_build(elem)