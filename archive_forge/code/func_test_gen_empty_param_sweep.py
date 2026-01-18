import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_gen_empty_param_sweep():
    out = params.sweep_from_proto(params_pb2.ParameterSweep())
    assert out == cirq.UnitSweep