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
def check_stdlibs(self) -> None:
    machine_choices = [MachineChoice.HOST]
    if self.coredata.is_cross_build():
        machine_choices.append(MachineChoice.BUILD)
    for for_machine in machine_choices:
        props = self.build.environment.properties[for_machine]
        for l in self.coredata.compilers[for_machine].keys():
            try:
                di = mesonlib.stringlistify(props.get_stdlib(l))
            except KeyError:
                continue
            if len(di) == 1:
                FeatureNew.single_use('stdlib without variable name', '0.56.0', self.subproject, location=self.current_node)
            kwargs = {'native': for_machine is MachineChoice.BUILD}
            name = l + '_stdlib'
            df = DependencyFallbacksHolder(self, [name])
            df.set_fallback(di)
            dep = df.lookup(kwargs, force_fallback=True)
            self.build.stdlibs[for_machine][l] = dep