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
@typed_pos_args('run_target', str)
@typed_kwargs('run_target', COMMAND_KW, DEPENDS_KW, ENV_KW.evolve(since='0.57.0'))
def func_run_target(self, node: mparser.FunctionNode, args: T.Tuple[str], kwargs: 'kwtypes.RunTarget') -> build.RunTarget:
    all_args = kwargs['command'].copy()
    for i in listify(all_args):
        if isinstance(i, ExternalProgram) and (not i.found()):
            raise InterpreterException(f'Tried to use non-existing executable {i.name!r}')
    if isinstance(all_args[0], str):
        all_args[0] = self.find_program_impl([all_args[0]])
    name = args[0]
    tg = build.RunTarget(name, all_args, kwargs['depends'], self.subdir, self.subproject, self.environment, kwargs['env'])
    self.add_target(name, tg)
    return tg