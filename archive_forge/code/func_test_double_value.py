import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_double_value():
    """Note: due to backwards compatibility, double_val conversion is one-way.
    double_val can be converted to python float,
    but a python float is converted into a float_val not a double_val.
    """
    msg = v2.program_pb2.Arg()
    msg.arg_value.double_value = 1.0
    parsed = arg_from_proto(msg, arg_function_language='')
    assert parsed == 1