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
@typed_pos_args('compiler.alignment', str)
@typed_kwargs('compiler.alignment', _PREFIX_KW, _ARGS_KW, _DEPENDENCIES_KW)
def alignment_method(self, args: T.Tuple[str], kwargs: 'AlignmentKw') -> int:
    typename = args[0]
    deps, msg = self._determine_dependencies(kwargs['dependencies'], compile_only=self.compiler.is_cross)
    result, cached = self.compiler.alignment(typename, kwargs['prefix'], self.environment, extra_args=kwargs['args'], dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    mlog.log('Checking for alignment of', mlog.bold(typename, True), msg, mlog.bold(str(result)), cached_msg)
    return result