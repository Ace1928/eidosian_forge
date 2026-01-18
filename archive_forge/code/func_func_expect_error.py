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
@typed_pos_args('expect_error', str)
@typed_kwargs('expect_error', KwargInfo('how', str, default='literal', validator=in_set_validator({'literal', 're'})))
def func_expect_error(self, node: mparser.BaseNode, args: T.Tuple[str], kwargs: TYPE_kwargs) -> ContextManagerObject:

    class ExpectErrorObject(ContextManagerObject):

        def __init__(self, msg: str, how: str, subproject: str) -> None:
            super().__init__(subproject)
            self.msg = msg
            self.how = how

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val is None:
                raise InterpreterException('Expecting an error but code block succeeded')
            if isinstance(exc_val, mesonlib.MesonException):
                msg = str(exc_val)
                if self.how == 'literal' and self.msg != msg or (self.how == 're' and (not re.match(self.msg, msg))):
                    raise InterpreterException(f'Expecting error {self.msg!r} but got {msg!r}')
                return True
    return ExpectErrorObject(args[0], kwargs['how'], self.subproject)