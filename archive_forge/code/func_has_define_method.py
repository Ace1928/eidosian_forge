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
@FeatureNew('compiler.has_define', '1.3.0')
@typed_pos_args('compiler.has_define', str)
@typed_kwargs('compiler.has_define', *_COMMON_KWS)
def has_define_method(self, args: T.Tuple[str], kwargs: 'CommonKW') -> bool:
    define_name = args[0]
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'], endl=None)
    value, cached = self.compiler.get_define(define_name, kwargs['prefix'], self.environment, extra_args=extra_args, dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    h = mlog.green('YES') if value is not None else mlog.red('NO')
    mlog.log('Checking if define', mlog.bold(define_name, True), msg, 'exists:', h, cached_msg)
    return value is not None