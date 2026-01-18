import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
def test_invalid_list():
    with pytest.raises(ValueError):
        _ = arg_to_proto(['', 1])
    with pytest.raises(ValueError):
        _ = arg_to_proto([1.0, ''])