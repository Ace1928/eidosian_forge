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
@typed_pos_args('import', str)
@typed_kwargs('import', REQUIRED_KW.evolve(since='0.59.0'), DISABLER_KW.evolve(since='0.59.0'))
@disablerIfNotFound
def func_import(self, node: mparser.BaseNode, args: T.Tuple[str], kwargs: 'kwtypes.FuncImportModule') -> T.Union[ExtensionModule, NewExtensionModule, NotFoundExtensionModule]:
    modname = args[0]
    disabled, required, _ = extract_required_kwarg(kwargs, self.subproject)
    if disabled:
        return NotFoundExtensionModule(modname)
    expect_unstable = False
    if modname.startswith(('unstable-', 'unstable_')):
        if modname.startswith('unstable_'):
            mlog.deprecation(f'Importing unstable modules as "{modname}" instead of "{modname.replace('_', '-', 1)}"', location=node)
        real_modname = modname[len('unstable') + 1:]
        expect_unstable = True
    else:
        real_modname = modname
    if real_modname in self.modules:
        return self.modules[real_modname]
    try:
        module = importlib.import_module(f'mesonbuild.modules.{real_modname}')
    except ImportError:
        if required:
            raise InvalidArguments(f'Module "{modname}" does not exist')
        ext_module = NotFoundExtensionModule(real_modname)
    else:
        ext_module = module.initialize(self)
        assert isinstance(ext_module, (ExtensionModule, NewExtensionModule))
        self.build.modules.append(real_modname)
    if ext_module.INFO.added:
        FeatureNew.single_use(f'module {ext_module.INFO.name}', ext_module.INFO.added, self.subproject, location=node)
    if ext_module.INFO.deprecated:
        FeatureDeprecated.single_use(f'module {ext_module.INFO.name}', ext_module.INFO.deprecated, self.subproject, location=node)
    if expect_unstable and (not ext_module.INFO.unstable) and (ext_module.INFO.stabilized is None):
        raise InvalidArguments(f'Module {ext_module.INFO.name} has never been unstable, remove "unstable-" prefix.')
    if ext_module.INFO.stabilized is not None:
        if expect_unstable:
            FeatureDeprecated.single_use(f'module {ext_module.INFO.name} has been stabilized', ext_module.INFO.stabilized, self.subproject, 'drop "unstable-" prefix from the module name', location=node)
        else:
            FeatureNew.single_use(f'module {ext_module.INFO.name} as stable module', ext_module.INFO.stabilized, self.subproject, f'Consider either adding "unstable-" to the module name, or updating the meson required version to ">= {ext_module.INFO.stabilized}"', location=node)
    elif ext_module.INFO.unstable:
        if not expect_unstable:
            if required:
                raise InvalidArguments(f'Module "{ext_module.INFO.name}" has not been stabilized, and must be imported as unstable-{ext_module.INFO.name}')
            ext_module = NotFoundExtensionModule(real_modname)
        else:
            mlog.warning(f'Module {ext_module.INFO.name} has no backwards or forwards compatibility and might not exist in future releases.', location=node, fatal=False)
    self.modules[real_modname] = ext_module
    return ext_module