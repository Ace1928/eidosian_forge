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
@noPosargs
@typed_kwargs('declare_dependency', KwargInfo('compile_args', ContainerTypeInfo(list, str), listify=True, default=[]), INCLUDE_DIRECTORIES.evolve(name='d_import_dirs', since='0.62.0'), D_MODULE_VERSIONS_KW.evolve(since='0.62.0'), KwargInfo('link_args', ContainerTypeInfo(list, str), listify=True, default=[]), DEPENDENCIES_KW, INCLUDE_DIRECTORIES, LINK_WITH_KW, LINK_WHOLE_KW.evolve(since='0.46.0'), DEPENDENCY_SOURCES_KW, KwargInfo('extra_files', ContainerTypeInfo(list, (mesonlib.File, str)), listify=True, default=[], since='1.2.0'), VARIABLES_KW.evolve(since='0.54.0', since_values={list: '0.56.0'}), KwargInfo('version', (str, NoneType)), KwargInfo('objects', ContainerTypeInfo(list, build.ExtractedObjects), listify=True, default=[], since='1.1.0'))
def func_declare_dependency(self, node: mparser.BaseNode, args: T.List[TYPE_var], kwargs: kwtypes.FuncDeclareDependency) -> dependencies.Dependency:
    deps = kwargs['dependencies']
    incs = self.extract_incdirs(kwargs)
    libs = kwargs['link_with']
    libs_whole = kwargs['link_whole']
    objects = kwargs['objects']
    sources = self.source_strings_to_files(kwargs['sources'])
    extra_files = self.source_strings_to_files(kwargs['extra_files'])
    compile_args = kwargs['compile_args']
    link_args = kwargs['link_args']
    variables = kwargs['variables']
    version = kwargs['version']
    if version is None:
        version = self.project_version
    d_module_versions = kwargs['d_module_versions']
    d_import_dirs = self.extract_incdirs(kwargs, 'd_import_dirs')
    srcdir = Path(self.environment.source_dir)
    for k, v in variables.items():
        if not v:
            FeatureNew.single_use('empty variable value in declare_dependency', '1.4.0', self.subproject, location=node)
        try:
            p = Path(v)
        except ValueError:
            continue
        else:
            if not self.is_subproject() and srcdir / self.subproject_dir in p.parents:
                continue
            if p.is_absolute() and p.is_dir() and (srcdir / self.root_subdir in [p] + list(Path(os.path.abspath(p)).parents)):
                variables[k] = P_OBJ.DependencyVariableString(v)
    dep = dependencies.InternalDependency(version, incs, compile_args, link_args, libs, libs_whole, sources, extra_files, deps, variables, d_module_versions, d_import_dirs, objects)
    return dep