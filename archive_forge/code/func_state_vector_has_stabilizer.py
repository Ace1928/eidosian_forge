from typing import Any, cast, Optional, Type
import numpy as np
from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.qis import clifford_tableau
from cirq.sim import state_vector_simulation_state, final_state_vector
from cirq.sim.clifford import (
def state_vector_has_stabilizer(state_vector: np.ndarray, stabilizer: DensePauliString) -> bool:
    """Checks that the state_vector is stabilized by the given stabilizer.

    The stabilizer should not modify the value of the state_vector, up to the
    global phase.

    Args:
        state_vector: An input state vector. Is not mutated by this function.
        stabilizer: A potential stabilizer of the above state_vector as a
          DensePauliString.

    Returns:
        Whether the stabilizer stabilizes the supplied state.
    """
    qubits = LineQubit.range(protocols.num_qubits(stabilizer))
    complex_dtype: Type[np.complexfloating] = np.complex64
    if np.issubdtype(state_vector.dtype, np.complexfloating):
        complex_dtype = cast(Type[np.complexfloating], state_vector.dtype)
    args = state_vector_simulation_state.StateVectorSimulationState(available_buffer=np.empty_like(state_vector), qubits=qubits, prng=np.random.RandomState(), initial_state=state_vector.copy(), dtype=complex_dtype)
    protocols.act_on(stabilizer, args, qubits)
    return np.allclose(args.target_tensor, state_vector)