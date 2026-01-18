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
@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CONFIG_TOOL, DependencyMethods.SYSTEM})
def curses_factory(env: 'Environment', for_machine: 'mesonlib.MachineChoice', kwargs: T.Dict[str, T.Any], methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    candidates: T.List['DependencyGenerator'] = []
    if DependencyMethods.PKGCONFIG in methods:
        pkgconfig_files = ['pdcurses', 'ncursesw', 'ncurses', 'curses']
        for pkg in pkgconfig_files:
            candidates.append(functools.partial(PkgConfigDependency, pkg, env, kwargs))
    if not env.machines[for_machine].is_windows():
        if DependencyMethods.CONFIG_TOOL in methods:
            candidates.append(functools.partial(CursesConfigToolDependency, 'curses', env, kwargs))
        if DependencyMethods.SYSTEM in methods:
            candidates.append(functools.partial(CursesSystemDependency, 'curses', env, kwargs))
    return candidates