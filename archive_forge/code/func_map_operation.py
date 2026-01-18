from typing import Callable
from cirq import circuits, devices, ops
def map_operation(self, operation: ops.Operation) -> ops.Operation:
    return operation.transform_qubits(self.qubit_map)