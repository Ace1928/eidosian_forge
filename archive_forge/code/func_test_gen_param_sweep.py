import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def test_gen_param_sweep():
    ps = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='foo', points=params_pb2.Points(points=[1, 2, 3]))]), params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='bar', points=params_pb2.Points(points=[4, 5]))])]))
    out = params.sweep_from_proto(ps)
    assert out == cirq.Product(cirq.Zip(cirq.Points('foo', [1, 2, 3])), cirq.Zip(cirq.Points('bar', [4, 5])))