from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@typed_kwargs('meson.override_dependency', NATIVE_KW, KwargInfo('static', (bool, NoneType), since='0.60.0'))
@typed_pos_args('meson.override_dependency', str, dependencies.Dependency)
@FeatureNew('meson.override_dependency', '0.54.0')
def override_dependency_method(self, args: T.Tuple[str, dependencies.Dependency], kwargs: 'FuncOverrideDependency') -> None:
    name, dep = args
    if not name:
        raise InterpreterException('First argument must be a string and cannot be empty')
    optkey = OptionKey('default_library', subproject=self.interpreter.subproject)
    default_library = self.interpreter.coredata.get_option(optkey)
    assert isinstance(default_library, str), 'for mypy'
    static = kwargs['static']
    if static is None:
        self._override_dependency_impl(name, dep, kwargs, static=None)
        if default_library == 'static':
            self._override_dependency_impl(name, dep, kwargs, static=True)
        elif default_library == 'shared':
            self._override_dependency_impl(name, dep, kwargs, static=False)
        else:
            self._override_dependency_impl(name, dep, kwargs, static=True)
            self._override_dependency_impl(name, dep, kwargs, static=False)
    else:
        self._override_dependency_impl(name, dep, kwargs, static=None, permissive=True)
        self._override_dependency_impl(name, dep, kwargs, static=static)