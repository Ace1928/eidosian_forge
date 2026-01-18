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
@FeatureNew('structured_sources', '0.62.0')
@typed_pos_args('structured_sources', object, optargs=[dict])
@noKwargs
@noArgsFlattening
def func_structured_sources(self, node: mparser.BaseNode, args: T.Tuple[object, T.Optional[T.Dict[str, object]]], kwargs: 'TYPE_kwargs') -> build.StructuredSources:
    valid_types = (str, mesonlib.File, build.GeneratedList, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)
    sources: T.Dict[str, T.List[T.Union[mesonlib.File, 'build.GeneratedTypes']]] = collections.defaultdict(list)
    for arg in mesonlib.listify(args[0]):
        if not isinstance(arg, valid_types):
            raise InvalidArguments(f'structured_sources: type "{type(arg)}" is not valid')
        if isinstance(arg, str):
            arg = mesonlib.File.from_source_file(self.environment.source_dir, self.subdir, arg)
        sources[''].append(arg)
    if args[1]:
        if '' in args[1]:
            raise InvalidArguments('structured_sources: keys to dictionary argument may not be an empty string.')
        for k, v in args[1].items():
            for arg in mesonlib.listify(v):
                if not isinstance(arg, valid_types):
                    raise InvalidArguments(f'structured_sources: type "{type(arg)}" is not valid')
                if isinstance(arg, str):
                    arg = mesonlib.File.from_source_file(self.environment.source_dir, self.subdir, arg)
                sources[k].append(arg)
    return build.StructuredSources(sources)