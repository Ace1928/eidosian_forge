from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
@typed_pos_args('hotdoc.generate_doc', str)
@typed_kwargs('hotdoc.generate_doc', KwargInfo('sitemap', file_types, required=True), KwargInfo('index', file_types, required=True), KwargInfo('project_version', str, required=True), KwargInfo('html_extra_theme', (str, NoneType)), KwargInfo('include_paths', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('dependencies', ContainerTypeInfo(list, (Dependency, build.StaticLibrary, build.SharedLibrary, CustomTarget, CustomTargetIndex)), listify=True, default=[]), KwargInfo('depends', ContainerTypeInfo(list, (CustomTarget, CustomTargetIndex)), listify=True, default=[], since='0.64.1'), KwargInfo('gi_c_source_roots', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('extra_assets', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('extra_extension_paths', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('subprojects', ContainerTypeInfo(list, HotdocTarget), listify=True, default=[]), KwargInfo('install', bool, default=False), allow_unknown=True)
def generate_doc(self, state: ModuleState, args: T.Tuple[str], kwargs: GenerateDocKwargs) -> ModuleReturnValue:
    project_name = args[0]
    if any((isinstance(x, (CustomTarget, CustomTargetIndex)) for x in kwargs['dependencies'])):
        FeatureDeprecated.single_use('hotdoc.generate_doc dependencies argument with custom_target', '0.64.1', state.subproject, 'use `depends`', state.current_node)
    builder = HotdocTargetBuilder(project_name, state, self.hotdoc, self.interpreter, kwargs)
    target, install_script = builder.make_targets()
    targets: T.List[T.Union[HotdocTarget, mesonlib.ExecutableSerialisation]] = [target]
    if install_script:
        targets.append(install_script)
    return ModuleReturnValue(target, targets)