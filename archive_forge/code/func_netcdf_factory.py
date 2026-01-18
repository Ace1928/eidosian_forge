from __future__ import annotations
import functools
import re
import typing as T
from .. import mesonlib
from .. import mlog
from .base import DependencyException, DependencyMethods
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency, CMakeDependencyFactory
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory, factory_methods
from .pkgconfig import PkgConfigDependency
@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CMAKE})
def netcdf_factory(env: 'Environment', for_machine: 'mesonlib.MachineChoice', kwargs: T.Dict[str, T.Any], methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    language = kwargs.get('language', 'c')
    if language not in ('c', 'cpp', 'fortran'):
        raise DependencyException(f'Language {language} is not supported with NetCDF.')
    candidates: T.List['DependencyGenerator'] = []
    if DependencyMethods.PKGCONFIG in methods:
        if language == 'fortran':
            pkg = 'netcdf-fortran'
        else:
            pkg = 'netcdf'
        candidates.append(functools.partial(PkgConfigDependency, pkg, env, kwargs, language=language))
    if DependencyMethods.CMAKE in methods:
        candidates.append(functools.partial(CMakeDependency, 'NetCDF', env, kwargs, language=language))
    return candidates