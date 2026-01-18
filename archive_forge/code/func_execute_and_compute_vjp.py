from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
@wraps(batch_execute_and_compute_vjp)
def execute_and_compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
    is_single_circuit = False
    if isinstance(circuits, QuantumScript):
        is_single_circuit = True
        circuits = [circuits]
        cotangents = [cotangents]
    results, vjps = batch_execute_and_compute_vjp(self, circuits, cotangents, execution_config)
    return (results[0], vjps[0]) if is_single_circuit else (results, vjps)