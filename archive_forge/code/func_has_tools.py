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
@FeatureNew('qt.has_tools', '0.54.0')
@noPosargs
@typed_kwargs('qt.has_tools', KwargInfo('required', (bool, coredata.UserFeatureOption), default=False), KwargInfo('method', str, default='auto'))
def has_tools(self, state: 'ModuleState', args: T.Tuple, kwargs: 'HasToolKwArgs') -> bool:
    method = kwargs.get('method', 'auto')
    disabled, required, feature = extract_required_kwarg(kwargs, state.subproject, default=False)
    if disabled:
        mlog.log('qt.has_tools skipped: feature', mlog.bold(feature), 'disabled')
        return False
    self._detect_tools(state, method, required=False)
    for tool in self.tools.values():
        if not tool.found():
            if required:
                raise MesonException('Qt tools not found')
            return False
    return True