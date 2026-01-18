from typing import Sequence
from typing import Union
import cirq
from cirq_ionq import ionq_gateset
def is_api_gate(self, operation: cirq.Operation) -> bool:
    return operation in self.gateset