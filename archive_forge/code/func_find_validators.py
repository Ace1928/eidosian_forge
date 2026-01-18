import math
import re
from collections import OrderedDict, deque
from collections.abc import Hashable as CollectionsHashable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from . import errors
from .datetime_parse import parse_date, parse_datetime, parse_duration, parse_time
from .typing import (
from .utils import almost_equal_floats, lenient_issubclass, sequence_like
def find_validators(type_: Type[Any], config: Type['BaseConfig']) -> Generator[AnyCallable, None, None]:
    from .dataclasses import is_builtin_dataclass, make_dataclass_validator
    if type_ is Any or type_ is object:
        return
    type_type = type_.__class__
    if type_type == ForwardRef or type_type == TypeVar:
        return
    if is_none_type(type_):
        yield none_validator
        return
    if type_ is Pattern or type_ is re.Pattern:
        yield pattern_validator
        return
    if type_ is Hashable or type_ is CollectionsHashable:
        yield hashable_validator
        return
    if is_callable_type(type_):
        yield callable_validator
        return
    if is_literal_type(type_):
        yield make_literal_validator(type_)
        return
    if is_builtin_dataclass(type_):
        yield from make_dataclass_validator(type_, config)
        return
    if type_ is Enum:
        yield enum_validator
        return
    if type_ is IntEnum:
        yield int_enum_validator
        return
    if is_namedtuple(type_):
        yield tuple_validator
        yield make_namedtuple_validator(type_, config)
        return
    if is_typeddict(type_):
        yield make_typeddict_validator(type_, config)
        return
    class_ = get_class(type_)
    if class_ is not None:
        if class_ is not Any and isinstance(class_, type):
            yield make_class_validator(class_)
        else:
            yield any_class_validator
        return
    for val_type, validators in _VALIDATORS:
        try:
            if issubclass(type_, val_type):
                for v in validators:
                    if isinstance(v, IfConfig):
                        if v.check(config):
                            yield v.validator
                    else:
                        yield v
                return
        except TypeError:
            raise RuntimeError(f'error checking inheritance of {type_!r} (type: {display_as_type(type_)})')
    if config.arbitrary_types_allowed:
        yield make_arbitrary_type_validator(type_)
    else:
        raise RuntimeError(f'no validator found for {type_}, see `arbitrary_types_allowed` in Config')