import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_invalid_float_arg():
    with pytest.raises(ValueError, match='unrecognized argument type'):
        _ = float_arg_from_proto(v2.program_pb2.Arg(arg_value=v2.program_pb2.ArgValue(float_value=0.5)), arg_function_language='test', required_arg_name='blah')