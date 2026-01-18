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
@typed_pos_args('compiler.has_multi_arguments', varargs=str)
@typed_kwargs('compiler.has_multi_arguments', _HAS_REQUIRED_KW)
@FeatureNew('compiler.has_multi_arguments', '0.37.0')
def has_multi_arguments_method(self, args: T.Tuple[T.List[str]], kwargs: 'HasArgumentKW') -> bool:
    return self._has_argument_impl(args[0], kwargs=kwargs)