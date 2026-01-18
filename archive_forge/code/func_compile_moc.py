from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
@FeatureNew('qt.compile_moc', '0.59.0')
@noPosargs
@typed_kwargs('qt.compile_moc', KwargInfo('sources', ContainerTypeInfo(list, (File, str, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)), listify=True, default=[]), KwargInfo('headers', ContainerTypeInfo(list, (File, str, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)), listify=True, default=[]), KwargInfo('extra_args', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('method', str, default='auto'), KwargInfo('include_directories', ContainerTypeInfo(list, (build.IncludeDirs, str)), listify=True, default=[]), KwargInfo('dependencies', ContainerTypeInfo(list, (Dependency, ExternalLibrary)), listify=True, default=[]), KwargInfo('preserve_paths', bool, default=False, since='1.4.0'))
def compile_moc(self, state: ModuleState, args: T.Tuple, kwargs: MocCompilerKwArgs) -> ModuleReturnValue:
    if any((isinstance(s, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)) for s in kwargs['headers'])):
        FeatureNew.single_use('qt.compile_moc: custom_target or generator for "headers" keyword argument', '0.60.0', state.subproject, location=state.current_node)
    if any((isinstance(s, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)) for s in kwargs['sources'])):
        FeatureNew.single_use('qt.compile_moc: custom_target or generator for "sources" keyword argument', '0.60.0', state.subproject, location=state.current_node)
    out = self._compile_moc_impl(state, kwargs)
    return ModuleReturnValue(out, [out])