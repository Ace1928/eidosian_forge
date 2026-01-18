import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def marganalize_counts(counts: Counts, qubit_index: Dict[int, int], qubits: Optional[List[int]]=None, clbits: Optional[List[int]]=None) -> np.ndarray:
    """Marginalization of the Counts. Verify that number of clbits equals to the number of qubits."""
    if clbits is not None:
        qubits_len = len(qubits) if not qubits is None else 0
        clbits_len = len(clbits) if not clbits is None else 0
        if clbits_len not in (0, qubits_len):
            raise QiskitError('Num qubits ({}) does not match number of clbits ({}).'.format(qubits_len, clbits_len))
        counts = marginal_counts(counts, clbits)
    if clbits is None and qubits is not None:
        clbits = [qubit_index[qubit] for qubit in qubits]
        counts = marginal_counts(counts, clbits)
    return counts