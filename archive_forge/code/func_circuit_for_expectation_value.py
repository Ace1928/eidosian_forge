import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def circuit_for_expectation_value(circuit: cirq.Circuit, pauli_string: cirq.PauliString) -> cirq.Circuit:
    """Sandwich a PauliString operator between a forwards and backwards
    copy of a circuit.

    This is a circuit representation of the expectation value of an operator
    <A> = <psi|A|psi> = <0|U^dag A U|0>. You can either extract the 0..0
    amplitude of the final state vector (assuming starting from the |0..0>
    state or extract the [0, 0] entry of the unitary matrix of this combined
    circuit.
    """
    assert pauli_string.coefficient == 1
    return cirq.Circuit([circuit, cirq.Moment((gate.on(q) for q, gate in pauli_string.items())), cirq.inverse(circuit)])