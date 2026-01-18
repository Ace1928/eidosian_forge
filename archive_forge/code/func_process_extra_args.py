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
def process_extra_args(self) -> None:
    for arg, value in self.kwargs.items():
        option = '--' + arg.replace('_', '-')
        self.check_extra_arg_type(arg, value)
        self.set_arg_value(option, value)