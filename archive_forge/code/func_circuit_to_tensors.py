import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def circuit_to_tensors(circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]]=None, initial_state: Union[int, None]=0) -> Tuple[List[qtn.Tensor], Dict['cirq.Qid', int], None]:
    """Given a circuit, construct a tensor network representation.

    Indices are named "i{i}_q{x}" where i is a time index and x is a
    qubit index.

    Args:
        circuit: The circuit containing operations that implement the
            cirq.unitary() protocol.
        qubits: A list of qubits in the circuit.
        initial_state: Either `0` corresponding to the |0..0> state, in
            which case the tensor network will represent the final
            state vector; or `None` in which case the starting indices
            will be left open and the tensor network will represent the
            circuit unitary.
    Returns:
        tensors: A list of quimb Tensor objects
        qubit_frontier: A mapping from qubit to time index at the end of
            the circuit. This can be used to deduce the names of the free
            tensor indices.
        positions: Currently None. May be changed in the future to return
            a suitable mapping for tn.graph()'s `fix` argument. Currently,
            `fix=None` will draw the resulting tensor network using a spring
            layout.

    Raises:
        ValueError: If the ihitial state is anything other than that
            corresponding to the |0> state.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    qubit_frontier = {q: 0 for q in qubits}
    positions = None
    tensors: List[qtn.Tensor] = []
    if initial_state == 0:
        for q in qubits:
            tensors += [qtn.Tensor(data=quimb.up().squeeze(), inds=(f'i0_q{q}',), tags={'Q0'})]
    elif initial_state is None:
        pass
    else:
        raise ValueError('Right now, only |0> or `None` initial states are supported.')
    for moment in circuit.moments:
        for op in moment.operations:
            assert cirq.has_unitary(op.gate)
            start_inds = [f'i{qubit_frontier[q]}_q{q}' for q in op.qubits]
            for q in op.qubits:
                qubit_frontier[q] += 1
            end_inds = [f'i{qubit_frontier[q]}_q{q}' for q in op.qubits]
            U = cirq.unitary(op).reshape((2,) * 2 * len(op.qubits))
            t = qtn.Tensor(data=U, inds=end_inds + start_inds, tags={f'Q{len(op.qubits)}'})
            tensors.append(t)
    return (tensors, qubit_frontier, positions)