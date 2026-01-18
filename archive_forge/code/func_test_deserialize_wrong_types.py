from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_wrong_types():
    serializer = cg.CircuitSerializer()
    proto = circuit_proto({'measurementgate': {'key': {'arg_value': {'float_value': 3.0}}, 'invert_mask': {'arg_value': {'bool_values': {'values': [True, False]}}}}, 'qubit_constant_index': [0]}, ['1_1'])
    with pytest.raises(ValueError, match='Incorrect types for measurement gate'):
        serializer.deserialize(proto)