from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@FeatureNew('meson.project_license()', '0.45.0')
@noPosargs
@noKwargs
def project_license_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> T.List[str]:
    return self.build.dep_manifest[self.interpreter.active_projectname].license