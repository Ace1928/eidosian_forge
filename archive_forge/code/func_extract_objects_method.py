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
@noKwargs
@typed_pos_args('extract_objects', varargs=(mesonlib.File, str, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList))
def extract_objects_method(self, args: T.Tuple[T.List[T.Union[mesonlib.FileOrString, 'build.GeneratedTypes']]], kwargs: TYPE_nkwargs) -> build.ExtractedObjects:
    return self._target_object.extract_objects(args[0])