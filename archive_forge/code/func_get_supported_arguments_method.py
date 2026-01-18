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
@FeatureNew('compiler.get_supported_arguments', '0.43.0')
@typed_pos_args('compiler.get_supported_arguments', varargs=str)
@typed_kwargs('compiler.get_supported_arguments', KwargInfo('checked', str, default='off', since='0.59.0', validator=in_set_validator({'warn', 'require', 'off'})))
def get_supported_arguments_method(self, args: T.Tuple[T.List[str]], kwargs: 'GetSupportedArgumentKw') -> T.List[str]:
    supported_args: T.List[str] = []
    checked = kwargs['checked']
    for arg in args[0]:
        if not self._has_argument_impl([arg]):
            msg = f'Compiler for {self.compiler.get_display_language()} does not support "{arg}"'
            if checked == 'warn':
                mlog.warning(msg)
            elif checked == 'require':
                raise mesonlib.MesonException(msg)
        else:
            supported_args.append(arg)
    return supported_args