import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def arg_to_proto(value: ARG_LIKE, *, arg_function_language: Optional[str]=None, out: Optional[v2.program_pb2.Arg]=None) -> v2.program_pb2.Arg:
    """Writes an argument value into an Arg proto.

    Args:
        value: The value to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.

    Returns:
        The proto that was written into as well as the `arg_function_language`
        that was used.

    Raises:
        ValueError: if the object holds unsupported values.
    """
    msg = v2.program_pb2.Arg() if out is None else out
    if isinstance(value, FLOAT_TYPES):
        msg.arg_value.float_value = float(value)
    elif isinstance(value, str):
        msg.arg_value.string_value = value
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value):
            if isinstance(value[0], str):
                if not all((isinstance(x, str) for x in value)):
                    raise ValueError('Sequences of mixed object types are not supported')
                msg.arg_value.string_values.values.extend((str(x) for x in value))
            else:
                numerical_fields = [[msg.arg_value.bool_values.values, (bool, np.bool_)], [msg.arg_value.int64_values.values, (int, np.integer, bool)], [msg.arg_value.double_values.values, (float, np.floating, int, bool)]]
                cur_index = 0
                non_numerical = None
                for v in value:
                    while cur_index < len(numerical_fields) and (not isinstance(v, numerical_fields[cur_index][1])):
                        cur_index += 1
                    if cur_index == len(numerical_fields):
                        non_numerical = v
                        break
                if non_numerical is not None:
                    raise ValueError(f'Mixed Sequences with objects of type {type(non_numerical)} are not supported')
                field, types_tuple = numerical_fields[cur_index]
                field.extend((types_tuple[0](x) for x in value))
    else:
        _arg_func_to_proto(value, arg_function_language, msg)
    return msg