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
@typed_pos_args('gnome.mkenums', str)
@typed_kwargs('gnome.mkenums', *_MK_ENUMS_COMMON_KWS, DEPENDS_KW, KwargInfo('sources', ContainerTypeInfo(list, (str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList)), listify=True, required=True), KwargInfo('c_template', (str, mesonlib.File, NoneType)), KwargInfo('h_template', (str, mesonlib.File, NoneType)), KwargInfo('comments', (str, NoneType)), KwargInfo('eprod', (str, NoneType)), KwargInfo('fhead', (str, NoneType)), KwargInfo('fprod', (str, NoneType)), KwargInfo('ftail', (str, NoneType)), KwargInfo('vhead', (str, NoneType)), KwargInfo('vprod', (str, NoneType)), KwargInfo('vtail', (str, NoneType)))
def mkenums(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'MkEnums') -> ModuleReturnValue:
    basename = args[0]
    c_template = kwargs['c_template']
    if isinstance(c_template, mesonlib.File):
        c_template = c_template.absolute_path(state.environment.source_dir, state.environment.build_dir)
    h_template = kwargs['h_template']
    if isinstance(h_template, mesonlib.File):
        h_template = h_template.absolute_path(state.environment.source_dir, state.environment.build_dir)
    cmd: T.List[str] = []
    known_kwargs = ['comments', 'eprod', 'fhead', 'fprod', 'ftail', 'identifier_prefix', 'symbol_prefix', 'vhead', 'vprod', 'vtail']
    for arg in known_kwargs:
        if kwargs[arg]:
            cmd += ['--' + arg.replace('_', '-'), kwargs[arg]]
    targets: T.List[CustomTarget] = []
    h_target: T.Optional[CustomTarget] = None
    if h_template is not None:
        h_output = os.path.basename(os.path.splitext(h_template)[0])
        h_cmd = cmd + ['--template', '@INPUT@']
        h_sources: T.List[T.Union[FileOrString, 'build.GeneratedTypes']] = [h_template]
        h_sources.extend(kwargs['sources'])
        h_target = self._make_mkenum_impl(state, h_sources, h_output, h_cmd, install=kwargs['install_header'], install_dir=kwargs['install_dir'])
        targets.append(h_target)
    if c_template is not None:
        c_output = os.path.basename(os.path.splitext(c_template)[0])
        c_cmd = cmd + ['--template', '@INPUT@']
        c_sources: T.List[T.Union[FileOrString, 'build.GeneratedTypes']] = [c_template]
        c_sources.extend(kwargs['sources'])
        depends = kwargs['depends'].copy()
        if h_target is not None:
            depends.append(h_target)
        c_target = self._make_mkenum_impl(state, c_sources, c_output, c_cmd, depends=depends)
        targets.insert(0, c_target)
    if c_template is None and h_template is None:
        generic_cmd = cmd + ['@INPUT@']
        target = self._make_mkenum_impl(state, kwargs['sources'], basename, generic_cmd, install=kwargs['install_header'], install_dir=kwargs['install_dir'])
        return ModuleReturnValue(target, [target])
    else:
        return ModuleReturnValue(targets, targets)