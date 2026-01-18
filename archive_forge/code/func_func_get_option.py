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
@typed_pos_args('get_option', str)
@noKwargs
def func_get_option(self, nodes: mparser.BaseNode, args: T.Tuple[str], kwargs: 'TYPE_kwargs') -> T.Union[coredata.UserOption, 'TYPE_var']:
    optname = args[0]
    if ':' in optname:
        raise InterpreterException('Having a colon in option name is forbidden, projects are not allowed to directly access options of other subprojects.')
    if optname_regex.search(optname.split('.', maxsplit=1)[-1]) is not None:
        raise InterpreterException(f'Invalid option name {optname!r}')
    opt = self.get_option_internal(optname)
    if isinstance(opt, coredata.UserFeatureOption):
        opt.name = optname
        return opt
    elif isinstance(opt, coredata.UserOption):
        if isinstance(opt.value, str):
            return P_OBJ.OptionString(opt.value, f'{{{optname}}}')
        return opt.value
    return opt