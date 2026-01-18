import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_empty_param_sweep_keys():
    assert params.sweep_from_proto(params_pb2.ParameterSweep()).keys == []