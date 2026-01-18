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
def find_program_impl(self, args: T.List[mesonlib.FileOrString], for_machine: MachineChoice=MachineChoice.HOST, default_options: T.Optional[T.Dict[OptionKey, T.Union[str, int, bool, T.List[str]]]]=None, required: bool=True, silent: bool=True, wanted: T.Union[str, T.List[str]]='', search_dirs: T.Optional[T.List[str]]=None, version_func: T.Optional[ProgramVersionFunc]=None) -> T.Union['ExternalProgram', 'build.Executable', 'OverrideProgram']:
    args = mesonlib.listify(args)
    extra_info: T.List[mlog.TV_Loggable] = []
    progobj = self.program_lookup(args, for_machine, default_options, required, search_dirs, wanted, version_func, extra_info)
    if progobj is None or not self.check_program_version(progobj, wanted, version_func, extra_info):
        progobj = self.notfound_program(args)
    if isinstance(progobj, ExternalProgram) and (not progobj.found()):
        if not silent:
            mlog.log('Program', mlog.bold(progobj.get_name()), 'found:', mlog.red('NO'), *extra_info)
        if required:
            m = 'Program {!r} not found or not executable'
            raise InterpreterException(m.format(progobj.get_name()))
        return progobj
    self.store_name_lookups(args)
    if not silent:
        mlog.log('Program', mlog.bold(progobj.name), 'found:', mlog.green('YES'), *extra_info)
    if isinstance(progobj, build.Executable):
        progobj.was_returned_by_find_program = True
    return progobj