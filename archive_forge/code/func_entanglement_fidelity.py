from typing import Optional, TYPE_CHECKING, Tuple
import numpy as np
from cirq import protocols, value, _import
from cirq.qis.states import (
def entanglement_fidelity(operation: 'cirq.SupportsKraus') -> float:
    """Returns entanglement fidelity of a given quantum channel.

    Entanglement fidelity $F_e$ of a quantum channel $E: L(H) \\to L(H)$ is the overlap between
    the maximally entangled state $|\\phi\\rangle = \\frac{1}{\\sqrt{dim H}} \\sum_i|i\\rangle|i\\rangle$
    and the state obtained by sending one half of $|\\phi\\rangle$ through the channel $E$, i.e.

        $$
        F_e = \\langle\\phi|(E \\otimes I)(|\\phi\\rangle\\langle\\phi|)|\\phi\\rangle
        $$

    where $I: L(H) \\to L(H)$ is the identity map.

    Args:
        operation: Quantum channel whose entanglement fidelity is to be computed.
    Returns:
        Entanglement fidelity of the channel represented by operation.
    """
    f = 0.0
    for k in protocols.kraus(operation):
        f += np.abs(np.trace(k)) ** 2
    n_qubits = protocols.num_qubits(operation)
    return float(f / 4 ** n_qubits)