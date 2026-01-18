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
@FeatureNew('dependency.as_link_whole', '0.56.0')
@noKwargs
@noPosargs
def as_link_whole_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> Dependency:
    if not isinstance(self.held_object, InternalDependency):
        raise InterpreterException('as_link_whole method is only supported on declare_dependency() objects')
    new_dep = self.held_object.generate_link_whole_dependency()
    return new_dep