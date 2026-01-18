from typing import cast, Dict, Iterable, List, Optional, Tuple
import sympy
import cirq
from cirq_google.api.v2 import batch_pb2
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter
def sweep_to_proto(sweep: cirq.Sweep, *, out: Optional[run_context_pb2.Sweep]=None) -> run_context_pb2.Sweep:
    """Converts a Sweep to v2 protobuf message.

    Args:
        sweep: The sweep to convert.
        out: Optional message to be populated. If not given, a new message will
            be created.

    Returns:
        Populated sweep protobuf message.

    Raises:
        ValueError: If the conversion cannot be completed successfully.
    """
    if out is None:
        out = run_context_pb2.Sweep()
    if sweep is cirq.UnitSweep:
        pass
    elif isinstance(sweep, cirq.Product):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.PRODUCT
        for factor in sweep.factors:
            sweep_to_proto(factor, out=out.sweep_function.sweeps.add())
    elif isinstance(sweep, cirq.Zip):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for s in sweep.sweeps:
            sweep_to_proto(s, out=out.sweep_function.sweeps.add())
    elif isinstance(sweep, cirq.Linspace) and (not isinstance(sweep.key, sympy.Expr)):
        out.single_sweep.parameter_key = sweep.key
        out.single_sweep.linspace.first_point = sweep.start
        out.single_sweep.linspace.last_point = sweep.stop
        out.single_sweep.linspace.num_points = sweep.length
        if sweep.metadata and getattr(sweep.metadata, 'path', None):
            out.single_sweep.parameter.path.extend(sweep.metadata.path)
        if sweep.metadata and getattr(sweep.metadata, 'idx', None):
            out.single_sweep.parameter.idx = sweep.metadata.idx
        if sweep.metadata and getattr(sweep.metadata, 'units', None):
            out.single_sweep.parameter.units = sweep.metadata.units
    elif isinstance(sweep, cirq.Points) and (not isinstance(sweep.key, sympy.Expr)):
        out.single_sweep.parameter_key = sweep.key
        out.single_sweep.points.points.extend(sweep.points)
        if sweep.metadata and getattr(sweep.metadata, 'path', None):
            out.single_sweep.parameter.path.extend(sweep.metadata.path)
        if sweep.metadata and getattr(sweep.metadata, 'idx', None):
            out.single_sweep.parameter.idx = sweep.metadata.idx
        if sweep.metadata and getattr(sweep.metadata, 'units', None):
            out.single_sweep.parameter.units = sweep.metadata.units
    elif isinstance(sweep, cirq.ListSweep):
        sweep_dict: Dict[str, List[float]] = {}
        for param_resolver in sweep:
            for key in param_resolver:
                if key not in sweep_dict:
                    sweep_dict[cast(str, key)] = []
                sweep_dict[cast(str, key)].append(cast(float, param_resolver.value_of(key)))
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for key in sweep_dict:
            sweep_to_proto(cirq.Points(key, sweep_dict[key]), out=out.sweep_function.sweeps.add())
    else:
        raise ValueError(f'cannot convert to v2 Sweep proto: {sweep}')
    return out