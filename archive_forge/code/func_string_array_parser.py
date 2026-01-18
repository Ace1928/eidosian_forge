from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
@typed_kwargs('string array option', KwargInfo('value', (ContainerTypeInfo(list, str), str, NoneType)), KwargInfo('choices', ContainerTypeInfo(list, str), default=[]))
def string_array_parser(self, description: str, args: T.Tuple[bool, _DEPRECATED_ARGS], kwargs: StringArrayArgs) -> coredata.UserOption:
    choices = kwargs['choices']
    value = kwargs['value'] if kwargs['value'] is not None else choices
    if isinstance(value, str):
        if value.startswith('['):
            FeatureDeprecated('String value for array option', '1.3.0').use(self.subproject)
        else:
            raise mesonlib.MesonException('Value does not define an array: ' + value)
    return coredata.UserArrayOption(description, value, choices=choices, yielding=args[0], deprecated=args[1])