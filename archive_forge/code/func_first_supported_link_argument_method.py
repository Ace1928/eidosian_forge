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
@FeatureNew('compiler.first_supported_link_argument_method', '0.46.0')
@noKwargs
@typed_pos_args('compiler.first_supported_link_argument', varargs=str)
def first_supported_link_argument_method(self, args: T.Tuple[T.List[str]], kwargs: 'TYPE_kwargs') -> T.List[str]:
    for arg in args[0]:
        if self._has_argument_impl([arg], mode=_TestMode.LINKER):
            mlog.log('First supported link argument:', mlog.bold(arg))
            return [arg]
    mlog.log('First supported link argument:', mlog.red('None'))
    return []