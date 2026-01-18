from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
@wraps(batch_compute_vjp)
def compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
    is_single_circuit = False
    if isinstance(circuits, QuantumScript):
        is_single_circuit = True
        circuits = [circuits]
        cotangents = [cotangents]
    res = batch_compute_vjp(self, circuits, cotangents, execution_config)
    return res[0] if is_single_circuit else res