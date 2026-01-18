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
def average_gate_fidelity(channel: QuantumChannel | Operator, target: Operator | None=None, require_cp: bool=True, require_tp: bool=False) -> float:
    """Return the average gate fidelity of a noisy quantum channel.

    The average gate fidelity :math:`F_{\\text{ave}}` is given by

    .. math::
        \\begin{aligned}
        F_{\\text{ave}}(\\mathcal{E}, U)
            &= \\int d\\psi \\langle\\psi|U^\\dagger
                \\mathcal{E}(|\\psi\\rangle\\!\\langle\\psi|)U|\\psi\\rangle \\\\
            &= \\frac{d F_{\\text{pro}}(\\mathcal{E}, U) + 1}{d + 1}
        \\end{aligned}

    where :math:`F_{\\text{pro}}(\\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.process_fidelity` of the input quantum
    *channel* :math:`\\mathcal{E}` with a *target* unitary :math:`U`, and
    :math:`d` is the dimension of the *channel*.

    Args:
        channel (QuantumChannel or Operator): noisy quantum channel.
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
        float: The average gate fidelity :math:`F_{\\text{ave}}`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
    """
    channel = _input_formatter(channel, SuperOp, 'average_gate_fidelity', 'channel')
    target = _input_formatter(target, Operator, 'average_gate_fidelity', 'target')
    if target is not None:
        try:
            target = Operator(target)
        except QiskitError as ex:
            raise QiskitError('Target channel is not a unitary channel. To compare two non-unitary channels use the `qiskit.quantum_info.process_fidelity` function instead.') from ex
    dim, _ = channel.dim
    f_pro = process_fidelity(channel, target=target, require_cp=require_cp, require_tp=require_tp)
    return (dim * f_pro + 1) / (dim + 1)