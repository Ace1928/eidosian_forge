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
@typed_pos_args('compiler.has_member', str, str)
@typed_kwargs('compiler.has_member', _HAS_REQUIRED_KW, *_COMMON_KWS)
def has_member_method(self, args: T.Tuple[str, str], kwargs: 'HasKW') -> bool:
    typename, membername = args
    disabled, required, feature = extract_required_kwarg(kwargs, self.subproject, default=False)
    if disabled:
        mlog.log('Type', mlog.bold(typename, True), 'has member', mlog.bold(membername, True), 'skipped: feature', mlog.bold(feature), 'disabled')
        return False
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'])
    had, cached = self.compiler.has_members(typename, [membername], kwargs['prefix'], self.environment, extra_args=extra_args, dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    if required and (not had):
        raise InterpreterException(f'{self.compiler.get_display_language()} member {membername!r} of type {typename!r} not usable')
    elif had:
        hadtxt = mlog.green('YES')
    else:
        hadtxt = mlog.red('NO')
    mlog.log('Checking whether type', mlog.bold(typename, True), 'has member', mlog.bold(membername, True), msg, hadtxt, cached_msg)
    return had