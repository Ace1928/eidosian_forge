from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
import os
import typing as T
from . import NewExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import build
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..coredata import BUILTIN_DIR_OPTIONS
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import D_MODULE_VERSIONS_KW, INSTALL_DIR_KW, VARIABLES_KW, NoneType
from ..interpreterbase import FeatureNew, FeatureDeprecated
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
def parse_variable_list(vardict: T.Dict[str, str]) -> T.List[T.Tuple[str, str]]:
    reserved = ['prefix', 'libdir', 'includedir']
    variables = []
    for name, value in vardict.items():
        if not value:
            FeatureNew.single_use('empty variable value in pkg.generate', '1.4.0', state.subproject, location=state.current_node)
        if not dataonly and name in reserved:
            raise mesonlib.MesonException(f'Variable "{name}" is reserved')
        variables.append((name, value))
    return variables