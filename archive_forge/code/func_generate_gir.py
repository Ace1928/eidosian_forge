from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@typed_pos_args('gnome.generate_gir', varargs=(Executable, build.SharedLibrary, build.StaticLibrary), min_varargs=1)
@typed_kwargs('gnome.generate_gir', INSTALL_KW, _BUILD_BY_DEFAULT.evolve(since='0.40.0'), _EXTRA_ARGS_KW, ENV_KW.evolve(since='1.2.0'), KwargInfo('dependencies', ContainerTypeInfo(list, Dependency), default=[], listify=True), KwargInfo('export_packages', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('fatal_warnings', bool, default=False, since='0.55.0'), KwargInfo('header', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('identifier_prefix', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('include_directories', ContainerTypeInfo(list, (str, build.IncludeDirs)), default=[], listify=True), KwargInfo('includes', ContainerTypeInfo(list, (str, GirTarget)), default=[], listify=True), KwargInfo('install_gir', (bool, NoneType), since='0.61.0'), KwargInfo('install_dir_gir', (str, bool, NoneType), deprecated_values={False: ('0.61.0', 'Use install_gir to disable installation')}, validator=lambda x: 'as boolean can only be false' if x is True else None), KwargInfo('install_typelib', (bool, NoneType), since='0.61.0'), KwargInfo('install_dir_typelib', (str, bool, NoneType), deprecated_values={False: ('0.61.0', 'Use install_typelib to disable installation')}, validator=lambda x: 'as boolean can only be false' if x is True else None), KwargInfo('link_with', ContainerTypeInfo(list, (build.SharedLibrary, build.StaticLibrary)), default=[], listify=True), KwargInfo('namespace', str, required=True), KwargInfo('nsversion', str, required=True), KwargInfo('sources', ContainerTypeInfo(list, (str, mesonlib.File, GeneratedList, CustomTarget, CustomTargetIndex)), default=[], listify=True), KwargInfo('symbol_prefix', ContainerTypeInfo(list, str), default=[], listify=True))
def generate_gir(self, state: 'ModuleState', args: T.Tuple[T.List[T.Union[Executable, build.SharedLibrary, build.StaticLibrary]]], kwargs: 'GenerateGir') -> ModuleReturnValue:
    state.add_language('c', MachineChoice.HOST)
    girtargets = [self._unwrap_gir_target(arg, state) for arg in args[0]]
    if len(girtargets) > 1 and any((isinstance(el, Executable) for el in girtargets)):
        raise MesonException('generate_gir only accepts a single argument when one of the arguments is an executable')
    gir_dep, giscanner, gicompiler = self._get_gir_dep(state)
    ns = kwargs['namespace']
    nsversion = kwargs['nsversion']
    libsources = kwargs['sources']
    girfile = f'{ns}-{nsversion}.gir'
    srcdir = os.path.join(state.environment.get_source_dir(), state.subdir)
    builddir = os.path.join(state.environment.get_build_dir(), state.subdir)
    depends: T.List[T.Union['FileOrString', 'build.GeneratedTypes', build.BuildTarget, build.StructuredSources]] = []
    depends.extend(gir_dep.sources)
    depends.extend(girtargets)
    langs_compilers = self._get_girtargets_langs_compilers(girtargets)
    cflags, internal_ldflags, external_ldflags = self._get_langs_compilers_flags(state, langs_compilers)
    deps = self._get_gir_targets_deps(girtargets)
    deps += kwargs['dependencies']
    deps += [gir_dep]
    typelib_includes, depends = self._gather_typelib_includes_and_update_depends(state, deps, depends)
    dep_cflags, dep_internal_ldflags, dep_external_ldflags, gi_includes, depends = self._get_dependencies_flags(deps, state, depends, use_gir_args=True)
    scan_cflags = []
    scan_cflags += list(self._get_scanner_cflags(cflags))
    scan_cflags += list(self._get_scanner_cflags(dep_cflags))
    scan_cflags += list(self._get_scanner_cflags(self._get_external_args_for_langs(state, [lc[0] for lc in langs_compilers])))
    scan_internal_ldflags = []
    scan_internal_ldflags += list(self._get_scanner_ldflags(internal_ldflags))
    scan_internal_ldflags += list(self._get_scanner_ldflags(dep_internal_ldflags))
    scan_external_ldflags = []
    scan_external_ldflags += list(self._get_scanner_ldflags(external_ldflags))
    scan_external_ldflags += list(self._get_scanner_ldflags(dep_external_ldflags))
    girtargets_inc_dirs = self._get_gir_targets_inc_dirs(girtargets)
    inc_dirs = kwargs['include_directories']
    gir_inc_dirs: T.List[str] = []
    scan_command: T.List[T.Union[str, Executable, 'ExternalProgram', 'OverrideProgram']] = [giscanner]
    scan_command += ['--quiet']
    scan_command += ['--no-libtool']
    scan_command += ['--namespace=' + ns, '--nsversion=' + nsversion]
    scan_command += ['--warn-all']
    scan_command += ['--output', '@OUTPUT@']
    scan_command += [f'--c-include={h}' for h in kwargs['header']]
    scan_command += kwargs['extra_args']
    scan_command += ['-I' + srcdir, '-I' + builddir]
    scan_command += state.get_include_args(girtargets_inc_dirs)
    scan_command += ['--filelist=' + self._make_gir_filelist(state, srcdir, ns, nsversion, girtargets, libsources)]
    for l in kwargs['link_with']:
        _cflags, depends = self._get_link_args(state, l, depends, use_gir_args=True)
        scan_command.extend(_cflags)
    _cmd, _ginc, _deps = self._scan_include(state, kwargs['includes'])
    scan_command.extend(_cmd)
    gir_inc_dirs.extend(_ginc)
    depends.extend(_deps)
    scan_command += [f'--symbol-prefix={p}' for p in kwargs['symbol_prefix']]
    scan_command += [f'--identifier-prefix={p}' for p in kwargs['identifier_prefix']]
    scan_command += [f'--pkg-export={p}' for p in kwargs['export_packages']]
    scan_command += ['--cflags-begin']
    scan_command += scan_cflags
    scan_command += ['--cflags-end']
    scan_command += state.get_include_args(inc_dirs)
    scan_command += state.get_include_args(itertools.chain(gi_includes, gir_inc_dirs, inc_dirs), prefix='--add-include-path=')
    scan_command += list(scan_internal_ldflags)
    scan_command += self._scan_gir_targets(state, girtargets)
    scan_command += self._scan_langs(state, [lc[0] for lc in langs_compilers])
    scan_command += list(scan_external_ldflags)
    if self._gir_has_option('--sources-top-dirs'):
        scan_command += ['--sources-top-dirs', os.path.join(state.environment.get_source_dir(), state.root_subdir)]
        scan_command += ['--sources-top-dirs', os.path.join(state.environment.get_build_dir(), state.root_subdir)]
    if '--warn-error' in scan_command:
        FeatureDeprecated.single_use('gnome.generate_gir argument --warn-error', '0.55.0', state.subproject, 'Use "fatal_warnings" keyword argument', state.current_node)
    if kwargs['fatal_warnings']:
        scan_command.append('--warn-error')
    generated_files = [f for f in libsources if isinstance(f, (GeneratedList, CustomTarget, CustomTargetIndex))]
    scan_target = self._make_gir_target(state, girfile, scan_command, generated_files, depends, T.cast('T.Dict[str, T.Any]', kwargs))
    typelib_output = f'{ns}-{nsversion}.typelib'
    typelib_cmd = [gicompiler, scan_target, '--output', '@OUTPUT@']
    typelib_cmd += state.get_include_args(gir_inc_dirs, prefix='--includedir=')
    for incdir in typelib_includes:
        typelib_cmd += ['--includedir=' + incdir]
    typelib_target = self._make_typelib_target(state, typelib_output, typelib_cmd, generated_files, T.cast('T.Dict[str, T.Any]', kwargs))
    self._devenv_prepend('GI_TYPELIB_PATH', os.path.join(state.environment.get_build_dir(), state.subdir))
    rv = [scan_target, typelib_target]
    return ModuleReturnValue(rv, rv)