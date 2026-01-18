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
@noPosargs
@typed_kwargs('python_installation.install_dir', _PURE_KW, _SUBDIR_KW)
def get_install_dir_method(self, args: T.List['TYPE_var'], kwargs: 'PyInstallKw') -> str:
    self.held_object.run_bytecompile[self.version] = True
    pure = kwargs['pure'] if kwargs['pure'] is not None else self.pure
    return self._get_install_dir_impl(pure, kwargs['subdir'])