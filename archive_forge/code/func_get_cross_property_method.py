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
@noArgsFlattening
@FeatureDeprecated('meson.get_cross_property', '0.58.0', 'Use meson.get_external_property() instead')
@typed_pos_args('meson.get_cross_property', str, optargs=[object])
@noKwargs
def get_cross_property_method(self, args: T.Tuple[str, T.Optional[object]], kwargs: 'TYPE_kwargs') -> object:
    propname, fallback = args
    return self.__get_external_property_impl(propname, fallback, MachineChoice.HOST)