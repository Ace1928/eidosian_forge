import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def float_arg_from_proto(arg_proto: v2.program_pb2.FloatArg, *, arg_function_language: str, required_arg_name: Optional[str]=None) -> Optional[FLOAT_ARG_LIKE]:
    """Extracts a python value from an argument value proto.

    This function handles `FloatArg` protos, that are required
    to be floats or symbolic expressions.

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
        ValueError: If the float arg proto is invalid.
    """
    which = arg_proto.WhichOneof('arg')
    if which == 'float_value':
        result = float(arg_proto.float_value)
        if round(result) == result:
            result = int(result)
        return result
    elif which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)
    elif which == 'func':
        func = _arg_func_from_proto(arg_proto.func, arg_function_language=arg_function_language, required_arg_name=required_arg_name)
        if func is None and required_arg_name is not None:
            raise ValueError(f'Arg {arg_proto.func} could not be processed for {required_arg_name}.')
        return cast(FLOAT_ARG_LIKE, func)
    elif which is None:
        if required_arg_name is not None:
            raise ValueError(f'Arg {required_arg_name} is missing.')
        return None
    else:
        raise ValueError(f'unrecognized argument type ({which}).')