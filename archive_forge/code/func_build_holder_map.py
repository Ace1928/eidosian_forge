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
def build_holder_map(self) -> None:
    """
            Build a mapping of `HoldableObject` types to their corresponding
            `ObjectHolder`s. This mapping is used in `InterpreterBase` to automatically
            holderify all returned values from methods and functions.
        """
    self.holder_map.update({list: P_OBJ.ArrayHolder, dict: P_OBJ.DictHolder, int: P_OBJ.IntegerHolder, bool: P_OBJ.BooleanHolder, str: P_OBJ.StringHolder, P_OBJ.MesonVersionString: P_OBJ.MesonVersionStringHolder, P_OBJ.DependencyVariableString: P_OBJ.DependencyVariableStringHolder, P_OBJ.OptionString: P_OBJ.OptionStringHolder, mesonlib.File: OBJ.FileHolder, build.SharedLibrary: OBJ.SharedLibraryHolder, build.StaticLibrary: OBJ.StaticLibraryHolder, build.BothLibraries: OBJ.BothLibrariesHolder, build.SharedModule: OBJ.SharedModuleHolder, build.Executable: OBJ.ExecutableHolder, build.Jar: OBJ.JarHolder, build.CustomTarget: OBJ.CustomTargetHolder, build.CustomTargetIndex: OBJ.CustomTargetIndexHolder, build.Generator: OBJ.GeneratorHolder, build.GeneratedList: OBJ.GeneratedListHolder, build.ExtractedObjects: OBJ.GeneratedObjectsHolder, build.RunTarget: OBJ.RunTargetHolder, build.AliasTarget: OBJ.AliasTargetHolder, build.Headers: OBJ.HeadersHolder, build.Man: OBJ.ManHolder, build.EmptyDir: OBJ.EmptyDirHolder, build.Data: OBJ.DataHolder, build.SymlinkData: OBJ.SymlinkDataHolder, build.InstallDir: OBJ.InstallDirHolder, build.IncludeDirs: OBJ.IncludeDirsHolder, mesonlib.EnvironmentVariables: OBJ.EnvironmentVariablesHolder, build.StructuredSources: OBJ.StructuredSourcesHolder, compilers.RunResult: compilerOBJ.TryRunResultHolder, dependencies.ExternalLibrary: OBJ.ExternalLibraryHolder, coredata.UserFeatureOption: OBJ.FeatureOptionHolder, envconfig.MachineInfo: OBJ.MachineHolder, build.ConfigurationData: OBJ.ConfigurationDataHolder})
    '\n            Build a mapping of `HoldableObject` base classes to their\n            corresponding `ObjectHolder`s. The difference to `self.holder_map`\n            is that the keys here define an upper bound instead of requiring an\n            exact match.\n\n            The mappings defined here are only used when there was no direct hit\n            found in `self.holder_map`.\n        '
    self.bound_holder_map.update({dependencies.Dependency: OBJ.DependencyHolder, ExternalProgram: OBJ.ExternalProgramHolder, compilers.Compiler: compilerOBJ.CompilerHolder, ModuleObject: OBJ.ModuleObjectHolder, MutableModuleObject: OBJ.MutableModuleObjectHolder})