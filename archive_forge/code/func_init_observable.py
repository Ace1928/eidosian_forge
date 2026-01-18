from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.bit import Bit
from qiskit.circuit.library.data_preparation import Initialize
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import PauliList, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
def init_observable(observable: BaseOperator | str) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        QiskitError: when observable type cannot be converted to SparsePauliOp.
    """
    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, BaseOperator) and (not isinstance(observable, BasePauli)):
        raise QiskitError(f'observable type not supported: {type(observable)}')
    else:
        if isinstance(observable, PauliList):
            raise QiskitError(f'observable type not supported: {type(observable)}')
        return SparsePauliOp(observable)