from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
@FeatureNew('subproject_options', '0.55.0')
@noKwargs
@noPosargs
def subproject_options(self, state: ModuleState, args: TYPE_var, kwargs: TYPE_kwargs) -> CMakeSubprojectOptions:
    return CMakeSubprojectOptions()