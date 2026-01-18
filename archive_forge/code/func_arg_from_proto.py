import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def arg_from_proto(arg_proto: v2.program_pb2.Arg, *, arg_function_language: str, required_arg_name: Optional[str]=None) -> Optional[ARG_RETURN_LIKE]:
    """Extracts a python value from an argument value proto.

    Args:
        arg_proto: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.
        required_arg_name: If set to `None`, the method will return `None` when
            given an unset proto value. If set to a string, the method will
            instead raise an error complaining that the value is missing in that
            situation.

    Returns:
        The deserialized value, or else None if there was no set value and
        `required_arg_name` was set to `None`.

    Raises:
        ValueError: If the arg protohas a value of an unrecognized type or is
            missing a required arg name.
    """
    which = arg_proto.WhichOneof('arg')
    if which == 'arg_value':
        arg_value = arg_proto.arg_value
        which_val = arg_value.WhichOneof('arg_value')
        if which_val == 'float_value' or which_val == 'double_value':
            if which_val == 'double_value':
                result = float(arg_value.double_value)
            else:
                result = float(arg_value.float_value)
            if math.ceil(result) == math.floor(result):
                result = int(result)
            return result
        if which_val == 'bool_values':
            return list(arg_value.bool_values.values)
        if which_val == 'string_value':
            return str(arg_value.string_value)
        if which_val == 'int64_values':
            return [int(v) for v in arg_value.int64_values.values]
        if which_val == 'double_values':
            return [float(v) for v in arg_value.double_values.values]
        if which_val == 'string_values':
            return [str(v) for v in arg_value.string_values.values]
        raise ValueError(f'Unrecognized value type: {which_val!r}')
    if which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)
    if which == 'func':
        func = _arg_func_from_proto(arg_proto.func, arg_function_language=arg_function_language, required_arg_name=required_arg_name)
        if func is not None:
            return func
    if required_arg_name is not None:
        raise ValueError(f'{required_arg_name} is missing or has an unrecognized argument type (WhichOneof("arg")={which!r}).')
    return None