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
@noPosargs
@typed_kwargs('extract_all_objects', KwargInfo('recursive', bool, default=False, since='0.46.0', not_set_warning=textwrap.dedent('                extract_all_objects called without setting recursive\n                keyword argument. Meson currently defaults to\n                non-recursive to maintain backward compatibility but\n                the default will be changed in the future.\n            ')))
def extract_all_objects_method(self, args: T.List[TYPE_nvar], kwargs: 'kwargs.BuildTargeMethodExtractAllObjects') -> build.ExtractedObjects:
    return self._target_object.extract_all_objects(kwargs['recursive'])