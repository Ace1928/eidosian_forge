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
@typed_pos_args('custom_target', optargs=[str])
@typed_kwargs('custom_target', COMMAND_KW, CT_BUILD_ALWAYS, CT_BUILD_ALWAYS_STALE, CT_BUILD_BY_DEFAULT, CT_INPUT_KW, CT_INSTALL_DIR_KW, CT_INSTALL_TAG_KW, MULTI_OUTPUT_KW, DEPENDS_KW, DEPEND_FILES_KW, DEPFILE_KW, ENV_KW.evolve(since='0.57.0'), INSTALL_KW, INSTALL_MODE_KW.evolve(since='0.47.0'), KwargInfo('feed', bool, default=False, since='0.59.0'), KwargInfo('capture', bool, default=False), KwargInfo('console', bool, default=False, since='0.48.0'))
def func_custom_target(self, node: mparser.FunctionNode, args: T.Tuple[str], kwargs: 'kwtypes.CustomTarget') -> build.CustomTarget:
    if kwargs['depfile'] and ('@BASENAME@' in kwargs['depfile'] or '@PLAINNAME@' in kwargs['depfile']):
        FeatureNew.single_use('substitutions in custom_target depfile', '0.47.0', self.subproject, location=node)
    install_mode = self._warn_kwarg_install_mode_sticky(kwargs['install_mode'])
    build_by_default = kwargs['build_by_default']
    build_always_stale = kwargs['build_always_stale']
    if kwargs['build_always'] is not None and kwargs['build_always_stale'] is not None:
        raise InterpreterException('CustomTarget: "build_always" and "build_always_stale" are mutually exclusive')
    if build_by_default is None and kwargs['install']:
        build_by_default = True
    elif kwargs['build_always'] is not None:
        if build_by_default is None:
            build_by_default = kwargs['build_always']
        build_always_stale = kwargs['build_by_default']
    if build_by_default is None:
        build_by_default = False
    if build_always_stale is None:
        build_always_stale = False
    name = args[0]
    if name is None:
        FeatureNew.single_use('custom_target() with no name argument', '0.60.0', self.subproject, location=node)
        name = ''
    inputs = self.source_strings_to_files(kwargs['input'], strict=False)
    command = kwargs['command']
    if command and isinstance(command[0], str):
        command[0] = self.find_program_impl([command[0]])
    if len(inputs) > 1 and kwargs['feed']:
        raise InvalidArguments('custom_target: "feed" keyword argument can only be used with a single input')
    if len(kwargs['output']) > 1 and kwargs['capture']:
        raise InvalidArguments('custom_target: "capture" keyword argument can only be used with a single output')
    if kwargs['capture'] and kwargs['console']:
        raise InvalidArguments('custom_target: "capture" and "console" keyword arguments are mutually exclusive')
    for c in command:
        if kwargs['capture'] and isinstance(c, str) and ('@OUTPUT@' in c):
            raise InvalidArguments('custom_target: "capture" keyword argument cannot be used with "@OUTPUT@"')
        if kwargs['feed'] and isinstance(c, str) and ('@INPUT@' in c):
            raise InvalidArguments('custom_target: "feed" keyword argument cannot be used with "@INPUT@"')
    if kwargs['install'] and (not kwargs['install_dir']):
        raise InvalidArguments('custom_target: "install_dir" keyword argument must be set when "install" is true.')
    if len(kwargs['install_dir']) > 1:
        FeatureNew.single_use('multiple install_dir for custom_target', '0.40.0', self.subproject, location=node)
    if len(kwargs['install_tag']) not in {0, 1, len(kwargs['output'])}:
        raise InvalidArguments(f'custom_target: install_tag argument must have 0 or 1 outputs, or the same number of elements as the output keyword argument. (there are {len(kwargs['install_tag'])} install_tags, and {len(kwargs['output'])} outputs)')
    for t in kwargs['output']:
        self.validate_forbidden_targets(t)
    self._validate_custom_target_outputs(len(inputs) > 1, kwargs['output'], 'custom_target')
    tg = build.CustomTarget(name, self.subdir, self.subproject, self.environment, command, inputs, kwargs['output'], build_always_stale=build_always_stale, build_by_default=build_by_default, capture=kwargs['capture'], console=kwargs['console'], depend_files=kwargs['depend_files'], depfile=kwargs['depfile'], extra_depends=kwargs['depends'], env=kwargs['env'], feed=kwargs['feed'], install=kwargs['install'], install_dir=kwargs['install_dir'], install_mode=install_mode, install_tag=kwargs['install_tag'], backend=self.backend)
    self.add_target(tg.name, tg)
    return tg