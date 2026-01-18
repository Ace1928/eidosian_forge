import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def merge_qubits(registers: Iterable[Register], **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]]) -> List[cirq.Qid]:
    """Merges the dictionary of appropriately shaped qubit arrays into a flat list of qubits."""
    ret: List[cirq.Qid] = []
    for reg in registers:
        if reg.name not in qubit_regs:
            raise ValueError(f'All qubit registers must be present. {reg.name} not in qubit_regs')
        qubits = qubit_regs[reg.name]
        qubits = np.array([qubits] if isinstance(qubits, cirq.Qid) else qubits)
        full_shape = reg.shape + (reg.bitsize,)
        if qubits.shape != full_shape:
            raise ValueError(f'{reg.name} register must of shape {full_shape} but is of shape {qubits.shape}')
        ret += qubits.flatten().tolist()
    return ret