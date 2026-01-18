import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def expr_builder(t: type, depth=0, direction=_FROM):

    def identity(expr):
        return expr
    origin = typing.get_origin(t)
    if origin == typing.Union:
        type_arg = typing.get_args(t)[0]
        inner = expr_builder(type_arg, depth + 1, direction)

        def f(expr):
            t0 = f'__{depth}'
            return f'{inner(t0)} if ({t0}:=({expr})) is not None else None'
        return f
    elif origin == list:
        type_arg = typing.get_args(t)[0]
        inner = expr_builder(type_arg, depth + 1, direction)

        def f(expr):
            t0 = f'__{depth}'
            return f'[{inner(t0)} for {t0} in {expr}]'
        return f
    elif origin == dict:
        key_type, value_type = typing.get_args(t)
        if not issubclass_safe(key_type, str):
            warnings.warn(f'to_json will not work for non-str key dict: {t}')
            return identity
        inner = expr_builder(value_type, depth + 1, direction)

        def f(expr):
            k0 = f'__k{depth}'
            v0 = f'__v{depth}'
            return '{' + f'{k0}: {inner(v0)} for {k0},{v0} in ({expr}).items()' + '}'
        return f
    elif is_dataclass(t):
        if not hasattr(t, '_lazyclasses_from_dict'):
            _process_class_internal(t)
        if direction == _FROM:

            def f(expr):
                return f'{t.__name__}._lazyclasses_from_dict({expr})'
            return f
        else:

            def f(expr):
                return f'({expr})._lazyclasses_to_dict()'
            return f
    elif issubclass_safe(t, Enum):
        if direction == _FROM:

            def f(expr):
                return f'{t.__name__}({expr})'
            return f
        else:

            def f(expr):
                return f'({expr}).value'
            return f
    from datetime import date, datetime
    if issubclass_safe(t, datetime):
        if direction == _FROM:

            def f(expr):
                t0 = f'__{depth}'
                return f'{t.__name__}.fromisoformat({t0}[:-1]+"+00:00" if ({t0}:={expr})[-1]=="Z" else {t0})'
            return f
        else:
            return lambda expr: f'({expr}).isoformat()'
    if issubclass_safe(t, date):
        if direction == _FROM:
            return lambda expr: f'{t.__name__}.fromisoformat({expr})'
        else:
            return lambda expr: f'({expr}).isoformat()'
    return identity