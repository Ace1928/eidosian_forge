import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_gen_param_sweep_zip():
    sweep = params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='foo', points=params_pb2.Points(points=[1, 2, 3])), params_pb2.SingleSweep(parameter_key='bar', points=params_pb2.Points(points=[4, 5]))])
    out = params._sweep_from_param_sweep_zip_proto(sweep)
    assert out == cirq.Points('foo', [1, 2, 3]) + cirq.Points('bar', [4, 5])