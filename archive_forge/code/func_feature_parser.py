from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
@typed_kwargs('feature option', KwargInfo('value', str, default='auto', validator=in_set_validator({'auto', 'enabled', 'disabled'})))
def feature_parser(self, description: str, args: T.Tuple[bool, _DEPRECATED_ARGS], kwargs: FeatureArgs) -> coredata.UserOption:
    return coredata.UserFeatureOption(description, kwargs['value'], *args)