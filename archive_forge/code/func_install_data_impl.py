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
def install_data_impl(self, sources: T.List[mesonlib.File], install_dir: str, install_mode: FileMode, rename: T.Optional[str], tag: T.Optional[str], install_data_type: T.Optional[str]=None, preserve_path: bool=False, follow_symlinks: T.Optional[bool]=None) -> build.Data:
    install_dir_name = install_dir.optname if isinstance(install_dir, P_OBJ.OptionString) else install_dir
    dirs = collections.defaultdict(list)
    if preserve_path:
        for file in sources:
            dirname = os.path.dirname(file.fname)
            dirs[dirname].append(file)
    else:
        dirs[''].extend(sources)
    ret_data = []
    for childdir, files in dirs.items():
        d = build.Data(files, os.path.join(install_dir, childdir), os.path.join(install_dir_name, childdir), install_mode, self.subproject, rename, tag, install_data_type, follow_symlinks)
        ret_data.append(d)
    self.build.data.extend(ret_data)
    return ret_data