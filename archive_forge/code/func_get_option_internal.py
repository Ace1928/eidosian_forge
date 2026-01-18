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
def get_option_internal(self, optname: str) -> coredata.UserOption:
    key = OptionKey.from_string(optname).evolve(subproject=self.subproject)
    if not key.is_project():
        for opts in [self.coredata.options, compilers.base_options]:
            v = opts.get(key)
            if v is None or v.yielding:
                v = opts.get(key.as_root())
            if v is not None:
                assert isinstance(v, coredata.UserOption), 'for mypy'
                return v
    try:
        opt = self.coredata.options[key]
        if opt.yielding and key.subproject and (key.as_root() in self.coredata.options):
            popt = self.coredata.options[key.as_root()]
            if type(opt) is type(popt):
                opt = popt
            else:
                opt_type = opt.__class__.__name__[4:][:-6].lower()
                popt_type = popt.__class__.__name__[4:][:-6].lower()
                mlog.warning('Option {0!r} of type {1!r} in subproject {2!r} cannot yield to parent option of type {3!r}, ignoring parent value. Use -D{2}:{0}=value to set the value for this option manually.'.format(optname, opt_type, self.subproject, popt_type), location=self.current_node)
        return opt
    except KeyError:
        pass
    raise InterpreterException(f'Tried to access unknown option {optname!r}.')