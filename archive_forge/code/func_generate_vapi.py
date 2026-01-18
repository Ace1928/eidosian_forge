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
@typed_pos_args('gnome.generate_vapi', str)
@typed_kwargs('gnome.generate_vapi', INSTALL_KW, INSTALL_DIR_KW, KwargInfo('sources', ContainerTypeInfo(list, (str, GirTarget), allow_empty=False), listify=True, required=True), KwargInfo('vapi_dirs', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('metadata_dirs', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('gir_dirs', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('packages', ContainerTypeInfo(list, (str, InternalDependency)), listify=True, default=[]))
def generate_vapi(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'GenerateVapi') -> ModuleReturnValue:
    created_values: T.List[T.Union[Dependency, build.Data]] = []
    library = args[0]
    build_dir = os.path.join(state.environment.get_build_dir(), state.subdir)
    source_dir = os.path.join(state.environment.get_source_dir(), state.subdir)
    pkg_cmd, vapi_depends, vapi_packages, vapi_includes, packages = self._extract_vapi_packages(state, kwargs['packages'])
    cmd: T.List[T.Union[ExternalProgram, Executable, OverrideProgram, str]]
    cmd = [state.find_program('vapigen'), '--quiet', f'--library={library}', f'--directory={build_dir}']
    cmd.extend([f'--vapidir={d}' for d in kwargs['vapi_dirs']])
    cmd.extend([f'--metadatadir={d}' for d in kwargs['metadata_dirs']])
    cmd.extend([f'--girdir={d}' for d in kwargs['gir_dirs']])
    cmd += pkg_cmd
    cmd += ['--metadatadir=' + source_dir]
    inputs = kwargs['sources']
    link_with: T.List[build.LibTypes] = []
    for i in inputs:
        if isinstance(i, str):
            cmd.append(os.path.join(source_dir, i))
        elif isinstance(i, GirTarget):
            link_with += self._get_vapi_link_with(i)
            subdir = os.path.join(state.environment.get_build_dir(), i.get_subdir())
            gir_file = os.path.join(subdir, i.get_outputs()[0])
            cmd.append(gir_file)
    vapi_output = library + '.vapi'
    datadir = state.environment.coredata.get_option(mesonlib.OptionKey('datadir'))
    assert isinstance(datadir, str), 'for mypy'
    install_dir = kwargs['install_dir'] or os.path.join(datadir, 'vala', 'vapi')
    if kwargs['install']:
        deps_target = self._generate_deps(state, library, vapi_packages, install_dir)
        created_values.append(deps_target)
    vapi_target = VapiTarget(vapi_output, state.subdir, state.subproject, state.environment, command=cmd, sources=inputs, outputs=[vapi_output], extra_depends=vapi_depends, install=kwargs['install'], install_dir=[install_dir], install_tag=['devel'])
    incs = [build.IncludeDirs(state.subdir, ['.'] + vapi_includes, False)]
    sources = [vapi_target] + vapi_depends
    rv = InternalDependency(None, incs, [], [], link_with, [], sources, [], [], {}, [], [], [])
    created_values.append(rv)
    return ModuleReturnValue(rv, created_values)