import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def split_qubits(registers: Iterable[Register], qubits: Sequence[cirq.Qid]) -> Dict[str, NDArray[cirq.Qid]]:
    """Splits the flat list of qubits into a dictionary of appropriately shaped qubit arrays."""
    qubit_regs = {}
    base = 0
    for reg in registers:
        qubit_regs[reg.name] = np.array(qubits[base:base + reg.total_bits()]).reshape(reg.shape + (reg.bitsize,))
        base += reg.total_bits()
    return qubit_regs