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
@typed_pos_args('generator', (build.Executable, ExternalProgram))
@typed_kwargs('generator', KwargInfo('arguments', ContainerTypeInfo(list, str, allow_empty=False), required=True, listify=True), KwargInfo('output', ContainerTypeInfo(list, str, allow_empty=False), required=True, listify=True), DEPFILE_KW, DEPENDS_KW, KwargInfo('capture', bool, default=False, since='0.43.0'))
def func_generator(self, node: mparser.FunctionNode, args: T.Tuple[T.Union[build.Executable, ExternalProgram]], kwargs: 'kwtypes.FuncGenerator') -> build.Generator:
    for rule in kwargs['output']:
        if '@BASENAME@' not in rule and '@PLAINNAME@' not in rule:
            raise InvalidArguments('Every element of "output" must contain @BASENAME@ or @PLAINNAME@.')
        if has_path_sep(rule):
            raise InvalidArguments('"output" must not contain a directory separator.')
    if len(kwargs['output']) > 1:
        for o in kwargs['output']:
            if '@OUTPUT@' in o:
                raise InvalidArguments('Tried to use @OUTPUT@ in a rule with more than one output.')
    gen = build.Generator(args[0], **kwargs)
    self.generators.append(gen)
    return gen