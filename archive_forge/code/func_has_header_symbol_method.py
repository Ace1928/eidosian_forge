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
@typed_pos_args('compiler.has_header_symbol', str, str)
@typed_kwargs('compiler.has_header_symbol', *_HEADER_KWS)
def has_header_symbol_method(self, args: T.Tuple[str, str], kwargs: 'HeaderKW') -> bool:
    hname, symbol = args
    disabled, required, feature = extract_required_kwarg(kwargs, self.subproject, default=False)
    if disabled:
        mlog.log('Header', mlog.bold(hname, True), 'has symbol', mlog.bold(symbol, True), 'skipped: feature', mlog.bold(feature), 'disabled')
        return False
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'])
    haz, cached = self.compiler.has_header_symbol(hname, symbol, kwargs['prefix'], self.environment, extra_args=extra_args, dependencies=deps)
    if required and (not haz):
        raise InterpreterException(f'{self.compiler.get_display_language()} symbol {symbol} not found in header {hname}')
    elif haz:
        h = mlog.green('YES')
    else:
        h = mlog.red('NO')
    cached_msg = mlog.blue('(cached)') if cached else ''
    mlog.log('Header', mlog.bold(hname, True), 'has symbol', mlog.bold(symbol, True), msg, h, cached_msg)
    return haz