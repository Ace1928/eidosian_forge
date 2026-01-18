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
def program_lookup(self, args: T.List[mesonlib.FileOrString], for_machine: MachineChoice, default_options: T.Optional[T.Dict[OptionKey, T.Union[str, int, bool, T.List[str]]]], required: bool, search_dirs: T.List[str], wanted: T.Union[str, T.List[str]], version_func: T.Optional[ProgramVersionFunc], extra_info: T.List[mlog.TV_Loggable]) -> T.Optional[T.Union[ExternalProgram, build.Executable, OverrideProgram]]:
    progobj = self.program_from_overrides(args, extra_info)
    if progobj:
        return progobj
    if args[0] == 'meson':
        return ExternalProgram('meson', self.environment.get_build_command(), silent=True)
    fallback = None
    wrap_mode = self.coredata.get_option(OptionKey('wrap_mode'))
    if wrap_mode != WrapMode.nofallback and self.environment.wrap_resolver:
        fallback = self.environment.wrap_resolver.find_program_provider(args)
    if fallback and wrap_mode == WrapMode.forcefallback:
        return self.find_program_fallback(fallback, args, default_options, required, extra_info)
    progobj = self.program_from_file_for(for_machine, args)
    if progobj is None:
        progobj = self.program_from_system(args, search_dirs, extra_info)
    if progobj is None and args[0].endswith('python3'):
        prog = ExternalProgram('python3', mesonlib.python_command, silent=True)
        progobj = prog if prog.found() else None
    if progobj and (not self.check_program_version(progobj, wanted, version_func, extra_info)):
        progobj = None
    if progobj is None and fallback and required:
        progobj = self.notfound_program(args)
        mlog.log('Program', mlog.bold(progobj.get_name()), 'found:', mlog.red('NO'), *extra_info)
        extra_info.clear()
        progobj = self.find_program_fallback(fallback, args, default_options, required, extra_info)
    return progobj