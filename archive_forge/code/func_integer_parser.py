from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
@typed_kwargs('integer option', KwargInfo('value', (int, str), default=True, deprecated_values={str: ('1.1.0', 'use an integer, not a string')}, convertor=int), KwargInfo('min', (int, NoneType)), KwargInfo('max', (int, NoneType)))
def integer_parser(self, description: str, args: T.Tuple[bool, _DEPRECATED_ARGS], kwargs: IntegerArgs) -> coredata.UserOption:
    value = kwargs['value']
    inttuple = (kwargs['min'], kwargs['max'], value)
    return coredata.UserIntegerOption(description, inttuple, *args)