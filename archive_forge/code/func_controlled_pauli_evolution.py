import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
def controlled_pauli_evolution(theta, wires, pauli_word, controls):
    """Controlled Evolution under generic Pauli words, adapted from the decomposition of
    qml.PauliRot to suit our needs


    Args:
        theta (float): rotation angle :math:`\\theta`
        pauli_word (string): the Pauli word defining the rotation
        wires (Iterable, Wires): the wires the operation acts on
        controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on

    Returns:
        list[Operator]: decomposition that make up the controlled evolution
    """
    active_wires, active_gates = zip(*[(wire, gate) for wire, gate in zip(wires, pauli_word) if gate != 'I'])
    ops = []
    for wire, gate in zip(active_wires, active_gates):
        if gate in ('X', 'Y'):
            ops.append(qml.Hadamard(wires=[wire]) if gate == 'X' else qml.RX(-np.pi / 2, wires=[wire]))
    ops.append(qml.CNOT(wires=[controls[1], wires[0]]))
    ops.append(qml.ctrl(op=qml.MultiRZ(theta, wires=list(active_wires)), control=controls[0]))
    ops.append(qml.CNOT(wires=[controls[1], wires[0]]))
    for wire, gate in zip(active_wires, active_gates):
        if gate in ('X', 'Y'):
            ops.append(qml.Hadamard(wires=[wire]) if gate == 'X' else qml.RX(-np.pi / 2, wires=[wire]))
    return ops