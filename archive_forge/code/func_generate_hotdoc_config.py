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
def generate_hotdoc_config(self) -> None:
    cwd = os.path.abspath(os.curdir)
    ncwd = os.path.join(self.sourcedir, self.subdir)
    mlog.log('Generating Hotdoc configuration for: ', mlog.bold(self.name))
    os.chdir(ncwd)
    if self.hotdoc.run_hotdoc(self.flatten_config_command()) != 0:
        raise MesonException('hotdoc failed to configure')
    os.chdir(cwd)