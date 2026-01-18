import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
@pytest.mark.parametrize('rotation_angles_arg', [{}, {'rotation_angles': [0.1, 0.3]}])
@pytest.mark.parametrize('qid_shape_arg', [{}, {'qid_shape': [2, 2]}])
@pytest.mark.parametrize('tags_arg', [{}, {'tags': ['test1', 'test2']}])
@pytest.mark.parametrize('lang', LANGUAGE_ORDER)
def test_internal_gate_serialization(rotation_angles_arg, qid_shape_arg, tags_arg, lang):
    g = cirq_google.InternalGate(gate_name='g', gate_module='test', num_qubits=5, **rotation_angles_arg, **qid_shape_arg, **tags_arg)
    proto = v2.program_pb2.InternalGate()
    internal_gate_arg_to_proto(g, out=proto)
    v = internal_gate_from_proto(proto, lang)
    assert g == v