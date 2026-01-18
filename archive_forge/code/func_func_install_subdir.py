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
@typed_pos_args('install_subdir', str)
@typed_kwargs('install_subdir', KwargInfo('install_dir', str, required=True), KwargInfo('strip_directory', bool, default=False), KwargInfo('exclude_files', ContainerTypeInfo(list, str), default=[], listify=True, since='0.42.0', validator=lambda x: 'cannot be absolute' if any((os.path.isabs(d) for d in x)) else None), KwargInfo('exclude_directories', ContainerTypeInfo(list, str), default=[], listify=True, since='0.42.0', validator=lambda x: 'cannot be absolute' if any((os.path.isabs(d) for d in x)) else None), INSTALL_MODE_KW.evolve(since='0.38.0'), INSTALL_TAG_KW.evolve(since='0.60.0'), INSTALL_FOLLOW_SYMLINKS)
def func_install_subdir(self, node: mparser.BaseNode, args: T.Tuple[str], kwargs: 'kwtypes.FuncInstallSubdir') -> build.InstallDir:
    exclude = (set(kwargs['exclude_files']), set(kwargs['exclude_directories']))
    srcdir = os.path.join(self.environment.source_dir, self.subdir, args[0])
    if not os.path.isdir(srcdir) or not any(os.listdir(srcdir)):
        FeatureNew.single_use('install_subdir with empty directory', '0.47.0', self.subproject, location=node)
        FeatureDeprecated.single_use('install_subdir with empty directory', '0.60.0', self.subproject, 'It worked by accident and is buggy. Use install_emptydir instead.', node)
    install_mode = self._warn_kwarg_install_mode_sticky(kwargs['install_mode'])
    idir_name = kwargs['install_dir']
    if isinstance(idir_name, P_OBJ.OptionString):
        idir_name = idir_name.optname
    idir = build.InstallDir(self.subdir, args[0], kwargs['install_dir'], idir_name, install_mode, exclude, kwargs['strip_directory'], self.subproject, install_tag=kwargs['install_tag'], follow_symlinks=kwargs['follow_symlinks'])
    self.build.install_dirs.append(idir)
    return idir