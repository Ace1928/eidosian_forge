from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_deserialize_fsim_missing_parameters():
    serializer = cg.CircuitSerializer()
    proto = circuit_proto({'fsimgate': {'theta': {'float_value': 3.0}}, 'qubit_constant_index': [0, 1]}, ['1_1', '1_2'])
    with pytest.raises(ValueError, match='theta and phi must be specified'):
        serializer.deserialize(proto)