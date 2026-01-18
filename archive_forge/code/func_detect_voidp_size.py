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
def detect_voidp_size(self, env: Environment) -> int:
    compilers = env.coredata.compilers.host
    compiler = compilers.get('c', None)
    if not compiler:
        compiler = compilers.get('cpp', None)
    if not compiler:
        raise mesonlib.MesonException('Requires a C or C++ compiler to compute sizeof(void *).')
    return compiler.sizeof('void *', '', env)[0]