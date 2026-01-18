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
def program_from_system(self, args: T.List[mesonlib.FileOrString], search_dirs: T.List[str], extra_info: T.List[mlog.TV_Loggable]) -> T.Optional[ExternalProgram]:
    source_dir = os.path.join(self.environment.get_source_dir(), self.subdir)
    for exename in args:
        if isinstance(exename, mesonlib.File):
            if exename.is_built:
                search_dir = os.path.join(self.environment.get_build_dir(), exename.subdir)
            else:
                search_dir = os.path.join(self.environment.get_source_dir(), exename.subdir)
            exename = exename.fname
            extra_search_dirs = []
        elif isinstance(exename, str):
            search_dir = source_dir
            extra_search_dirs = search_dirs
        else:
            raise InvalidArguments(f'find_program only accepts strings and files, not {exename!r}')
        extprog = ExternalProgram(exename, search_dir=search_dir, extra_search_dirs=extra_search_dirs, silent=True)
        if extprog.found():
            extra_info.append(f'({' '.join(extprog.get_command())})')
            return extprog
    return None