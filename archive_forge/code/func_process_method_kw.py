from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
def process_method_kw(possible: T.Iterable[DependencyMethods], kwargs: T.Dict[str, T.Any]) -> T.List[DependencyMethods]:
    method: T.Union[DependencyMethods, str] = kwargs.get('method', 'auto')
    if isinstance(method, DependencyMethods):
        return [method]
    if method not in [e.value for e in DependencyMethods]:
        raise DependencyException(f'method {method!r} is invalid')
    method = DependencyMethods(method)
    if method is DependencyMethods.CONFIG_TOOL:
        pass
    if method in [DependencyMethods.SDLCONFIG, DependencyMethods.CUPSCONFIG, DependencyMethods.PCAPCONFIG, DependencyMethods.LIBWMFCONFIG]:
        method = DependencyMethods.CONFIG_TOOL
    if method is DependencyMethods.QMAKE:
        method = DependencyMethods.CONFIG_TOOL
    if method == DependencyMethods.AUTO:
        methods = list(possible)
    elif method in possible:
        methods = [method]
    else:
        raise DependencyException('Unsupported detection method: {}, allowed methods are {}'.format(method.value, mlog.format_list([x.value for x in [DependencyMethods.AUTO] + list(possible)])))
    return methods