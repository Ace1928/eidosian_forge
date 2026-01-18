from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
@typed_kwargs('boolean option', KwargInfo('value', (bool, str), default=True, validator=lambda x: None if isinstance(x, bool) or x in {'true', 'false'} else 'boolean options must have boolean values', deprecated_values={str: ('1.1.0', 'use a boolean, not a string')}))
def boolean_parser(self, description: str, args: T.Tuple[bool, _DEPRECATED_ARGS], kwargs: BooleanArgs) -> coredata.UserOption:
    return coredata.UserBooleanOption(description, kwargs['value'], *args)