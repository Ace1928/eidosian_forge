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
@typed_pos_args('run_command', (build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str), varargs=(build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str))
@typed_kwargs('run_command', KwargInfo('check', (bool, NoneType), since='0.47.0'), KwargInfo('capture', bool, default=True, since='0.47.0'), ENV_KW.evolve(since='0.50.0'))
def func_run_command(self, node: mparser.BaseNode, args: T.Tuple[T.Union[build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str], T.List[T.Union[build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str]]], kwargs: 'kwtypes.RunCommand') -> RunProcess:
    return self.run_command_impl(args, kwargs)