from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
@wraps(batch_compute_jvp)
def compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
    is_single_circuit = False
    if isinstance(circuits, QuantumScript):
        is_single_circuit = True
        circuits = [circuits]
        tangents = [tangents]
    res = batch_compute_jvp(self, circuits, tangents, execution_config)
    return res[0] if is_single_circuit else res