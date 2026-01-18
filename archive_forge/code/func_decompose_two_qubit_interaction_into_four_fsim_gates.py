from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def decompose_two_qubit_interaction_into_four_fsim_gates(interaction: Union['cirq.SupportsUnitary', np.ndarray], *, fsim_gate: Union['cirq.FSimGate', 'cirq.ISwapPowGate'], qubits: Optional[Sequence['cirq.Qid']]=None) -> 'cirq.Circuit':
    """Decomposes operations into an FSimGate near theta=pi/2, phi=0.

    This decomposition is guaranteed to use exactly four of the given FSim
    gates. It works by decomposing into two B gates and then decomposing each
    B gate into two of the given FSim gate.

    This decomposition only works for FSim gates with a theta (iswap angle)
    between 3/8π and 5/8π (i.e. within 22.5° of maximum strength) and a
    phi (cphase angle) between -π/4 and +π/4 (i.e. within 45° of minimum
    strength).

    Args:
        interaction: The two qubit operation to synthesize. This can either be
            a cirq object (such as a gate, operation, or circuit) or a raw numpy
            array specifying the 4x4 unitary matrix.
        fsim_gate: The only two qubit gate that is permitted to appear in the
            output. Must satisfy 3/8π < phi < 5/8π and abs(theta) < pi/4.
        qubits: The qubits that the resulting operations should apply the
            desired interaction to. If not set then defaults to either the
            qubits of the given interaction (if it is a `cirq.Operation`) or
            else to `cirq.LineQubit.range(2)`.

    Returns:
        A list of operations implementing the desired two qubit unitary. The
        list will include four operations of the given fsim gate, various single
        qubit operations, and a global phase operation.

    Raises:
        ValueError: If the `fsim_gate` has invalid angles or is parameterized, or
            if the supplied target to synthesize acts on more than two qubits.
    """
    if protocols.is_parameterized(fsim_gate):
        raise ValueError('FSimGate must not have parameterized values for angles.')
    if isinstance(fsim_gate, ops.ISwapPowGate):
        mapped_gate = ops.FSimGate(-fsim_gate.exponent * np.pi / 2, 0)
    else:
        mapped_gate = fsim_gate
    theta, phi = (mapped_gate.theta, mapped_gate.phi)
    assert isinstance(theta, float) and isinstance(phi, float)
    if not 3 / 8 * np.pi <= abs(theta) <= 5 / 8 * np.pi:
        raise ValueError('Must have 3π/8 ≤ |fsim_gate.theta| ≤ 5π/8')
    if abs(phi) > np.pi / 4:
        raise ValueError('Must have abs(fsim_gate.phi) ≤ π/4')
    if qubits is None:
        if isinstance(interaction, ops.Operation):
            qubits = interaction.qubits
        else:
            qubits = devices.LineQubit.range(2)
    if len(qubits) != 2:
        raise ValueError(f'Expected a pair of qubits, but got {qubits!r}.')
    kak = linalg.kak_decomposition(interaction)
    result_using_b_gates = _decompose_two_qubit_interaction_into_two_b_gates(kak, qubits=qubits)
    b_decomposition = _decompose_b_gate_into_two_fsims(fsim_gate=mapped_gate, qubits=qubits)
    b_decomposition = [fsim_gate(*op.qubits) if op.gate == mapped_gate else op for op in b_decomposition]
    result = circuits.Circuit()
    for op in result_using_b_gates:
        if isinstance(op.gate, _BGate):
            result.append(b_decomposition)
        else:
            result.append(op)
    return result