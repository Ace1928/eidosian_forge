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
@noPosargs
@noKwargs
@FeatureDeprecated('meson.source_root', '0.56.0', 'use meson.project_source_root() or meson.global_source_root() instead.')
def source_root_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> str:
    return self.interpreter.environment.source_dir