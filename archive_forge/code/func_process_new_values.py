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
def process_new_values(self, invalues: T.List[T.Union[TYPE_var, ExecutableSerialisation]]) -> None:
    invalues = listify(invalues)
    for v in invalues:
        if isinstance(v, ObjectHolder):
            raise InterpreterException('Modules must not return ObjectHolders')
        if isinstance(v, (build.BuildTarget, build.CustomTarget, build.RunTarget)):
            self.add_target(v.name, v)
        elif isinstance(v, list):
            self.process_new_values(v)
        elif isinstance(v, ExecutableSerialisation):
            v.subproject = self.subproject
            self.build.install_scripts.append(v)
        elif isinstance(v, build.Data):
            self.build.data.append(v)
        elif isinstance(v, build.SymlinkData):
            self.build.symlinks.append(v)
        elif isinstance(v, dependencies.InternalDependency):
            self.process_new_values(v.sources[0])
        elif isinstance(v, build.InstallDir):
            self.build.install_dirs.append(v)
        elif isinstance(v, Test):
            self.build.tests.append(v)
        elif isinstance(v, (int, str, bool, Disabler, ObjectHolder, build.GeneratedList, ExternalProgram, build.ConfigurationData)):
            pass
        else:
            raise InterpreterException(f'Module returned a value of unknown type {v!r}.')