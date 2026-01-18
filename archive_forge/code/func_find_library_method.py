from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
@disablerIfNotFound
@typed_pos_args('compiler.find_library', str)
@typed_kwargs('compiler.find_library', KwargInfo('required', (bool, coredata.UserFeatureOption), default=True), KwargInfo('has_headers', ContainerTypeInfo(list, str), listify=True, default=[], since='0.50.0'), KwargInfo('static', (bool, NoneType), since='0.51.0'), KwargInfo('disabler', bool, default=False, since='0.49.0'), KwargInfo('dirs', ContainerTypeInfo(list, str), listify=True, default=[]), *(k.evolve(name=f'header_{k.name}') for k in _HEADER_KWS))
def find_library_method(self, args: T.Tuple[str], kwargs: 'FindLibraryKW') -> 'dependencies.ExternalLibrary':
    libname = args[0]
    disabled, required, feature = extract_required_kwarg(kwargs, self.subproject)
    if disabled:
        mlog.log('Library', mlog.bold(libname), 'skipped: feature', mlog.bold(feature), 'disabled')
        return self.notfound_library(libname)
    has_header_kwargs: 'HeaderKW' = {'required': required, 'args': kwargs['header_args'], 'dependencies': kwargs['header_dependencies'], 'include_directories': kwargs['header_include_directories'], 'prefix': kwargs['header_prefix'], 'no_builtin_args': kwargs['header_no_builtin_args']}
    for h in kwargs['has_headers']:
        if not self._has_header_impl(h, has_header_kwargs):
            return self.notfound_library(libname)
    search_dirs = extract_search_dirs(kwargs)
    prefer_static = self.environment.coredata.get_option(OptionKey('prefer_static'))
    if kwargs['static'] is True:
        libtype = mesonlib.LibType.STATIC
    elif kwargs['static'] is False:
        libtype = mesonlib.LibType.SHARED
    elif prefer_static:
        libtype = mesonlib.LibType.PREFER_STATIC
    else:
        libtype = mesonlib.LibType.PREFER_SHARED
    linkargs = self.compiler.find_library(libname, self.environment, search_dirs, libtype)
    if required and (not linkargs):
        if libtype == mesonlib.LibType.PREFER_SHARED:
            libtype_s = 'shared or static'
        else:
            libtype_s = libtype.name.lower()
        raise InterpreterException('{} {} library {!r} not found'.format(self.compiler.get_display_language(), libtype_s, libname))
    lib = dependencies.ExternalLibrary(libname, linkargs, self.environment, self.compiler.language)
    return lib