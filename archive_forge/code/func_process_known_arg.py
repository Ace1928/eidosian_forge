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
def process_known_arg(self, option: str, argname: T.Optional[str]=None, value_processor: T.Optional[T.Callable]=None) -> None:
    if not argname:
        argname = option.strip('-').replace('-', '_')
    value = self.kwargs.pop(argname)
    if value is not None and value_processor:
        value = value_processor(value)
    self.set_arg_value(option, value)