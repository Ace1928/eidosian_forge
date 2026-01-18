from __future__ import annotations
import functools
import typing as T
import os
import re
from ..environment import detect_cpu_family
from .base import DependencyMethods, detect_compiler, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import factory_methods
from .pkgconfig import PkgConfigDependency
@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CONFIG_TOOL, DependencyMethods.SYSTEM})
def mpi_factory(env: 'Environment', for_machine: 'MachineChoice', kwargs: T.Dict[str, T.Any], methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    language = kwargs.get('language', 'c')
    if language not in {'c', 'cpp', 'fortran'}:
        return []
    candidates: T.List['DependencyGenerator'] = []
    compiler = detect_compiler('mpi', env, for_machine, language)
    if not compiler:
        return []
    compiler_is_intel = compiler.get_id() in {'intel', 'intel-cl'}
    if DependencyMethods.PKGCONFIG in methods and (not compiler_is_intel):
        pkg_name = None
        if language == 'c':
            pkg_name = 'ompi-c'
        elif language == 'cpp':
            pkg_name = 'ompi-cxx'
        elif language == 'fortran':
            pkg_name = 'ompi-fort'
        candidates.append(functools.partial(PkgConfigDependency, pkg_name, env, kwargs, language=language))
    if DependencyMethods.CONFIG_TOOL in methods:
        nwargs = kwargs.copy()
        if compiler_is_intel:
            if env.machines[for_machine].is_windows():
                nwargs['version_arg'] = '-v'
                nwargs['returncode_value'] = 3
            if language == 'c':
                tool_names = [os.environ.get('I_MPI_CC'), 'mpiicc']
            elif language == 'cpp':
                tool_names = [os.environ.get('I_MPI_CXX'), 'mpiicpc']
            elif language == 'fortran':
                tool_names = [os.environ.get('I_MPI_F90'), 'mpiifort']
            cls: T.Type[ConfigToolDependency] = IntelMPIConfigToolDependency
        else:
            if language == 'c':
                tool_names = [os.environ.get('MPICC'), 'mpicc']
            elif language == 'cpp':
                tool_names = [os.environ.get('MPICXX'), 'mpic++', 'mpicxx', 'mpiCC']
            elif language == 'fortran':
                tool_names = [os.environ.get(e) for e in ['MPIFC', 'MPIF90', 'MPIF77']]
                tool_names.extend(['mpifort', 'mpif90', 'mpif77'])
            cls = OpenMPIConfigToolDependency
        tool_names = [t for t in tool_names if t]
        assert tool_names
        nwargs['tools'] = tool_names
        candidates.append(functools.partial(cls, tool_names[0], env, nwargs, language=language))
    if DependencyMethods.SYSTEM in methods:
        candidates.append(functools.partial(MSMPIDependency, 'msmpi', env, kwargs, language=language))
    return candidates