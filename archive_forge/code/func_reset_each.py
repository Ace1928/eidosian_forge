import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
def reset_each(*qubits: 'cirq.Qid') -> List[raw_types.Operation]:
    """Returns a list of `cirq.ResetChannel` instances on the given qubits."""
    return [ResetChannel(q.dimension).on(q) for q in qubits]