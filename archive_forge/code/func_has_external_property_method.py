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
@FeatureNew('meson.has_external_property', '0.58.0')
@typed_pos_args('meson.has_external_property', str)
@typed_kwargs('meson.has_external_property', NATIVE_KW)
def has_external_property_method(self, args: T.Tuple[str], kwargs: 'NativeKW') -> bool:
    prop_name = args[0]
    return prop_name in self.interpreter.environment.properties[kwargs['native']]