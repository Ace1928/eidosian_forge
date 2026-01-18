from __future__ import annotations
import sysconfig
import typing as T
from .. import mesonlib
from . import ExtensionModule, ModuleInfo, ModuleState
from ..build import (
from ..interpreter.type_checking import SHARED_MOD_KWS
from ..interpreterbase import typed_kwargs, typed_pos_args, noPosargs, noKwargs, permittedKwargs
from ..programs import ExternalProgram
@noKwargs
@typed_pos_args('python3.sysconfig_path', str)
def sysconfig_path(self, state, args, kwargs):
    path_name = args[0]
    valid_names = sysconfig.get_path_names()
    if path_name not in valid_names:
        raise mesonlib.MesonException(f'{path_name} is not a valid path name {valid_names}.')
    return sysconfig.get_path(path_name, vars={'base': '', 'platbase': '', 'installed_base': ''})[1:]