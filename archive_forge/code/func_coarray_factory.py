from __future__ import annotations
import functools
import typing as T
from .base import DependencyMethods, detect_compiler, SystemDependency
from .cmake import CMakeDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .factory import factory_methods
@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CMAKE, DependencyMethods.SYSTEM})
def coarray_factory(env: 'Environment', for_machine: 'MachineChoice', kwargs: T.Dict[str, T.Any], methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    fcid = detect_compiler('coarray', env, for_machine, 'fortran').get_id()
    candidates: T.List['DependencyGenerator'] = []
    if fcid == 'gcc':
        if DependencyMethods.PKGCONFIG in methods:
            for pkg in ['caf-openmpi', 'caf']:
                candidates.append(functools.partial(PkgConfigDependency, pkg, env, kwargs, language='fortran'))
        if DependencyMethods.CMAKE in methods:
            if 'modules' not in kwargs:
                kwargs['modules'] = 'OpenCoarrays::caf_mpi'
            candidates.append(functools.partial(CMakeDependency, 'OpenCoarrays', env, kwargs, language='fortran'))
    if DependencyMethods.SYSTEM in methods:
        candidates.append(functools.partial(CoarrayDependency, env, kwargs))
    return candidates