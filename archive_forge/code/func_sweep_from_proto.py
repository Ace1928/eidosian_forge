from typing import cast, Dict, Iterable, List, Optional, Tuple
import sympy
import cirq
from cirq_google.api.v2 import batch_pb2
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter
def sweep_from_proto(msg: run_context_pb2.Sweep) -> cirq.Sweep:
    """Creates a Sweep from a v2 protobuf message."""
    which = msg.WhichOneof('sweep')
    if which is None:
        return cirq.UnitSweep
    if which == 'sweep_function':
        factors = [sweep_from_proto(m) for m in msg.sweep_function.sweeps]
        func_type = msg.sweep_function.function_type
        if func_type == run_context_pb2.SweepFunction.PRODUCT:
            return cirq.Product(*factors)
        if func_type == run_context_pb2.SweepFunction.ZIP:
            return cirq.Zip(*factors)
        raise ValueError(f'invalid sweep function type: {func_type}')
    if which == 'single_sweep':
        key = msg.single_sweep.parameter_key
        if msg.single_sweep.HasField('parameter'):
            metadata = DeviceParameter(path=msg.single_sweep.parameter.path, idx=msg.single_sweep.parameter.idx if msg.single_sweep.parameter.HasField('idx') else None, units=msg.single_sweep.parameter.units if msg.single_sweep.parameter.HasField('units') else None)
        else:
            metadata = None
        if msg.single_sweep.WhichOneof('sweep') == 'linspace':
            return cirq.Linspace(key=key, start=msg.single_sweep.linspace.first_point, stop=msg.single_sweep.linspace.last_point, length=msg.single_sweep.linspace.num_points, metadata=metadata)
        if msg.single_sweep.WhichOneof('sweep') == 'points':
            return cirq.Points(key=key, points=msg.single_sweep.points.points, metadata=metadata)
        raise ValueError(f'single sweep type not set: {msg}')
    raise ValueError(f'sweep type not set: {msg}')