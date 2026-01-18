from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
@typed_pos_args('python_installation.get_path', str, optargs=[object])
@noKwargs
def get_path_method(self, args: T.Tuple[str, T.Optional['TYPE_var']], kwargs: 'TYPE_kwargs') -> 'TYPE_var':
    path_name, fallback = args
    try:
        return self.paths[path_name]
    except KeyError:
        if fallback is not None:
            return fallback
        raise InvalidArguments(f'{path_name} is not a valid path name')