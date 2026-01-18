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
def generate_rust_target(self, target: build.BuildTarget) -> None:
    rustc = target.compilers['rust']
    base_proxy = target.get_options()
    args = rustc.compiler_args()
    args += compilers.get_base_compile_args(base_proxy, rustc)
    self.generate_generator_list_rules(target)
    deps: T.List[str] = []
    project_deps: T.List[RustDep] = []
    orderdeps: T.List[str] = []
    main_rust_file = None
    if target.structured_sources:
        if target.structured_sources.needs_copy():
            _ods, main_rust_file = self.__generate_sources_structure(Path(self.get_target_private_dir(target)) / 'structured', target.structured_sources)
            orderdeps.extend(_ods)
        else:
            g = target.structured_sources.first_file()
            if isinstance(g, File):
                main_rust_file = g.rel_to_builddir(self.build_to_src)
            elif isinstance(g, GeneratedList):
                main_rust_file = os.path.join(self.get_target_private_dir(target), g.get_outputs()[0])
            else:
                main_rust_file = os.path.join(g.get_subdir(), g.get_outputs()[0])
            for f in target.structured_sources.as_list():
                if isinstance(f, File):
                    orderdeps.append(f.rel_to_builddir(self.build_to_src))
                else:
                    orderdeps.extend([os.path.join(self.build_to_src, f.subdir, s) for s in f.get_outputs()])
    for i in target.get_sources():
        if not rustc.can_compile(i):
            raise InvalidArguments(f'Rust target {target.get_basename()} contains a non-rust source file.')
        if main_rust_file is None:
            main_rust_file = i.rel_to_builddir(self.build_to_src)
    for g in target.get_generated_sources():
        for i in g.get_outputs():
            if not rustc.can_compile(i):
                raise InvalidArguments(f'Rust target {target.get_basename()} contains a non-rust source file.')
            if isinstance(g, GeneratedList):
                fname = os.path.join(self.get_target_private_dir(target), i)
            else:
                fname = os.path.join(g.get_subdir(), i)
            if main_rust_file is None:
                main_rust_file = fname
            orderdeps.append(fname)
    if main_rust_file is None:
        raise RuntimeError('A Rust target has no Rust sources. This is weird. Also a bug. Please report')
    target_name = os.path.join(target.subdir, target.get_filename())
    cratetype = target.rust_crate_type
    args.extend(['--crate-type', cratetype])
    if cratetype in {'bin', 'dylib'}:
        args.extend(rustc.get_linker_always_args())
    args += self.generate_basic_compiler_args(target, rustc)
    args += ['--crate-name', target.name.replace('-', '_').replace(' ', '_').replace('.', '_')]
    depfile = os.path.join(target.subdir, target.name + '.d')
    args += ['--emit', f'dep-info={depfile}', '--emit', f'link={target_name}']
    args += ['--out-dir', self.get_target_private_dir(target)]
    args += ['-C', 'metadata=' + target.get_id()]
    args += target.get_extra_args('rust')
    if not isinstance(target, build.StaticLibrary):
        try:
            buildtype = target.get_option(OptionKey('buildtype'))
            crt = target.get_option(OptionKey('b_vscrt'))
            args += rustc.get_crt_link_args(crt, buildtype)
        except KeyError:
            pass
    if mesonlib.version_compare(rustc.version, '>= 1.67.0'):
        verbatim = '+verbatim'
    else:
        verbatim = ''

    def _link_library(libname: str, static: bool, bundle: bool=False):
        type_ = 'static' if static else 'dylib'
        modifiers = []
        if not bundle and static:
            modifiers.append('-bundle')
        if verbatim:
            modifiers.append(verbatim)
        if modifiers:
            type_ += ':' + ','.join(modifiers)
        args.append(f'-l{type_}={libname}')
    linkdirs = mesonlib.OrderedSet()
    external_deps = target.external_deps.copy()
    target_deps = target.get_dependencies()
    for d in target_deps:
        linkdirs.add(d.subdir)
        deps.append(self.get_dependency_filename(d))
        if isinstance(d, build.StaticLibrary):
            external_deps.extend(d.external_deps)
        if d.uses_rust_abi():
            if d not in itertools.chain(target.link_targets, target.link_whole_targets):
                continue
            d_name = self._get_rust_dependency_name(target, d)
            args += ['--extern', '{}={}'.format(d_name, os.path.join(d.subdir, d.filename))]
            project_deps.append(RustDep(d_name, self.rust_crates[d.name].order))
            continue
        lib = self.get_target_filename_for_linking(d)
        link_whole = d in target.link_whole_targets
        if isinstance(target, build.StaticLibrary) or (isinstance(target, build.Executable) and rustc.get_crt_static()):
            static = isinstance(d, build.StaticLibrary)
            libname = os.path.basename(lib) if verbatim else d.name
            _link_library(libname, static, bundle=link_whole)
        elif link_whole:
            link_whole_args = rustc.linker.get_link_whole_for([lib])
            args += [f'-Clink-arg={a}' for a in link_whole_args]
        else:
            args.append(f'-Clink-arg={lib}')
    for e in external_deps:
        for a in e.get_link_args():
            if a in rustc.native_static_libs:
                pass
            elif a.startswith('-L'):
                args.append(a)
            elif a.endswith(('.dll', '.so', '.dylib', '.a', '.lib')) and isinstance(target, build.StaticLibrary):
                dir_, lib = os.path.split(a)
                linkdirs.add(dir_)
                if not verbatim:
                    lib, ext = os.path.splitext(lib)
                    if lib.startswith('lib'):
                        lib = lib[3:]
                static = a.endswith(('.a', '.lib'))
                _link_library(lib, static)
            else:
                args.append(f'-Clink-arg={a}')
    for d in linkdirs:
        d = d or '.'
        args.append(f'-L{d}')
    args.extend((f'-Clink-arg={a}' for a in target.get_used_stdlib_args('rust')))
    has_shared_deps = any((isinstance(dep, build.SharedLibrary) for dep in target_deps))
    has_rust_shared_deps = any((dep.uses_rust() and dep.rust_crate_type == 'dylib' for dep in target_deps))
    if cratetype in {'dylib', 'proc-macro'} or has_rust_shared_deps:
        args += ['-C', 'prefer-dynamic']
    if isinstance(target, build.SharedLibrary) or has_shared_deps:
        if has_path_sep(target.name):
            target_slashname_workaround_dir = os.path.join(os.path.dirname(target.name), self.get_target_dir(target))
        else:
            target_slashname_workaround_dir = self.get_target_dir(target)
        rpath_args, target.rpath_dirs_to_remove = rustc.build_rpath_args(self.environment, self.environment.get_build_dir(), target_slashname_workaround_dir, self.determine_rpath_dirs(target), target.build_rpath, target.install_rpath)
        for rpath_arg in rpath_args:
            args += ['-C', 'link-arg=' + rpath_arg + ':' + os.path.join(rustc.get_sysroot(), 'lib')]
    proc_macro_dylib_path = None
    if getattr(target, 'rust_crate_type', '') == 'proc-macro':
        proc_macro_dylib_path = os.path.abspath(os.path.join(target.subdir, target.get_filename()))
    self._add_rust_project_entry(target.name, os.path.abspath(os.path.join(self.environment.build_dir, main_rust_file)), args, bool(target.subproject), proc_macro_dylib_path, project_deps)
    compiler_name = self.compiler_to_rule_name(rustc)
    element = NinjaBuildElement(self.all_outputs, target_name, compiler_name, main_rust_file)
    if orderdeps:
        element.add_orderdep(orderdeps)
    if deps:
        element.add_dep(deps)
    element.add_item('ARGS', args)
    element.add_item('targetdep', depfile)
    element.add_item('cratetype', cratetype)
    self.add_build(element)
    if isinstance(target, build.SharedLibrary):
        self.generate_shsym(target)
    self.create_target_source_introspection(target, rustc, args, [main_rust_file], [])