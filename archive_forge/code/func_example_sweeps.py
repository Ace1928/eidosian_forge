import pytest
import cirq
import cirq_google.api.v1.params as params
from cirq_google.api.v1 import params_pb2
def example_sweeps():
    empty_sweep = params_pb2.ParameterSweep()
    empty_product = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep())
    empty_zip = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(), params_pb2.ZipSweep()]))
    full_sweep = params_pb2.ParameterSweep(sweep=params_pb2.ProductSweep(factors=[params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='11', linspace=params_pb2.Linspace(first_point=0, last_point=10, num_points=5)), params_pb2.SingleSweep(parameter_key='12', points=params_pb2.Points(points=range(7)))]), params_pb2.ZipSweep(sweeps=[params_pb2.SingleSweep(parameter_key='21', linspace=params_pb2.Linspace(first_point=0, last_point=10, num_points=11)), params_pb2.SingleSweep(parameter_key='22', points=params_pb2.Points(points=range(13)))])]))
    return [empty_sweep, empty_product, empty_zip, full_sweep]