from __future__ import annotations
import logging
import numpy as np
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel import Choi, SuperOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.utils import optionals as _optionals
def gate_error(channel: QuantumChannel, target: Operator | None=None, require_cp: bool=True, require_tp: bool=False) -> float:
    """Return the gate error of a noisy quantum channel.

    The gate error :math:`E` is given by the average gate infidelity

    .. math::
        E(\\mathcal{E}, U) = 1 - F_{\\text{ave}}(\\mathcal{E}, U)

    where :math:`F_{\\text{ave}}(\\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.average_gate_fidelity` of the input
    quantum *channel* :math:`\\mathcal{E}` with a *target* unitary
    :math:`U`.

    Args:
        channel (QuantumChannel): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): check if input and target channels are
                           completely-positive and if non-CP log warning
                           containing negative eigenvalues of Choi-matrix
                           [Default: True].
        require_tp (bool): check if input and target channels are
                           trace-preserving and if non-TP log warning
                           containing negative eigenvalues of partial
                           Choi-matrix :math:`Tr_{\\text{out}}[\\mathcal{E}] - I`
                           [Default: True].

    Returns:
        float: The average gate error :math:`E`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
    """
    channel = _input_formatter(channel, SuperOp, 'gate_error', 'channel')
    target = _input_formatter(target, Operator, 'gate_error', 'target')
    return 1 - average_gate_fidelity(channel, target=target, require_cp=require_cp, require_tp=require_tp)