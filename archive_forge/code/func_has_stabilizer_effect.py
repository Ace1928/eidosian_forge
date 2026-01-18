from typing import Any, Optional
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq._import import LazyLoader
import cirq.protocols.unitary_protocol as unitary_protocol
import cirq.protocols.has_unitary_protocol as has_unitary_protocol
import cirq.protocols.qid_shape_protocol as qid_shape_protocol
import cirq.protocols.decompose_protocol as decompose_protocol
def has_stabilizer_effect(val: Any) -> bool:
    """Returns whether the input has a stabilizer effect.

    For 1-qubit gates always returns correct result. For other operations relies
    on the operation to define whether it has stabilizer effect.
    """
    strats = [_strat_has_stabilizer_effect_from_has_stabilizer_effect, _strat_has_stabilizer_effect_from_gate, _strat_has_stabilizer_effect_from_unitary, _strat_has_stabilizer_effect_from_decompose]
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result
    return False