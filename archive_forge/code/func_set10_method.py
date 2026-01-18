from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
@typed_pos_args('configuration_data.set10', str, (int, bool))
@typed_kwargs('configuration_data.set10', _CONF_DATA_SET_KWS)
def set10_method(self, args: T.Tuple[str, T.Union[int, bool]], kwargs: 'kwargs.ConfigurationDataSet') -> None:
    self.__check_used()
    if not isinstance(args[1], bool):
        mlog.deprecation('configuration_data.set10 with number. the `set10` method should only be used with booleans', location=self.interpreter.current_node)
        if args[1] < 0:
            mlog.warning('Passing a number that is less than 0 may not have the intended result, as meson will treat all non-zero values as true.', location=self.interpreter.current_node)
    self.held_object.values[args[0]] = (int(args[1]), kwargs['description'])