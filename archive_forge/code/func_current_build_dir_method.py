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
def current_build_dir_method(self, args: T.List['TYPE_var'], kwargs: 'TYPE_kwargs') -> str:
    src = self.interpreter.environment.build_dir
    sub = self.interpreter.subdir
    if sub == '':
        return src
    return os.path.join(src, sub)