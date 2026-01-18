from typing import Dict, Iterable, TYPE_CHECKING
from cirq import ops
import cirq.contrib.acquaintance as cca
def final_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
    mapping = dict(self.initial_mapping)
    cca.update_mapping(mapping, self.circuit.all_operations())
    return mapping