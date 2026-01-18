from __future__ import annotations
from .. import mparser
from .. import environment
from .. import coredata
from .. import dependencies
from .. import mlog
from .. import build
from .. import optinterpreter
from .. import compilers
from .. import envconfig
from ..wrap import wrap, WrapMode
from .. import mesonlib
from ..mesonlib import (EnvironmentVariables, ExecutableSerialisation, MesonBugException, MesonException, HoldableObject,
from ..programs import ExternalProgram, NonExistingExternalProgram
from ..dependencies import Dependency
from ..depfile import DepFile
from ..interpreterbase import ContainerTypeInfo, InterpreterBase, KwargInfo, typed_kwargs, typed_pos_args
from ..interpreterbase import noPosargs, noKwargs, permittedKwargs, noArgsFlattening, noSecondLevelHolderResolving, unholder_return
from ..interpreterbase import InterpreterException, InvalidArguments, InvalidCode, SubdirDoneRequest
from ..interpreterbase import Disabler, disablerIfNotFound
from ..interpreterbase import FeatureNew, FeatureDeprecated, FeatureBroken, FeatureNewKwargs
from ..interpreterbase import ObjectHolder, ContextManagerObject
from ..interpreterbase import stringifyUserArguments
from ..modules import ExtensionModule, ModuleObject, MutableModuleObject, NewExtensionModule, NotFoundExtensionModule
from ..optinterpreter import optname_regex
from . import interpreterobjects as OBJ
from . import compiler as compilerOBJ
from .mesonmain import MesonMain
from .dependencyfallbacks import DependencyFallbacksHolder
from .interpreterobjects import (
from .type_checking import (
from . import primitives as P_OBJ
from pathlib import Path
from enum import Enum
import os
import shutil
import uuid
import re
import stat
import collections
import typing as T
import textwrap
import importlib
import copy
def check_program_version(self, progobj: T.Union[ExternalProgram, build.Executable, OverrideProgram], wanted: T.Union[str, T.List[str]], version_func: T.Optional[ProgramVersionFunc], extra_info: T.List[mlog.TV_Loggable]) -> bool:
    if wanted:
        if version_func:
            version = version_func(progobj)
        elif isinstance(progobj, build.Executable):
            if progobj.subproject:
                interp = self.subprojects[progobj.subproject].held_object
            else:
                interp = self
            assert isinstance(interp, Interpreter)
            version = interp.project_version
        else:
            version = progobj.get_version(self)
        is_found, not_found, _ = mesonlib.version_compare_many(version, wanted)
        if not is_found:
            extra_info[:0] = ['found', mlog.normal_cyan(version), 'but need:', mlog.bold(', '.join([f"'{e}'" for e in not_found]))]
            return False
        extra_info.insert(0, mlog.normal_cyan(version))
    return True