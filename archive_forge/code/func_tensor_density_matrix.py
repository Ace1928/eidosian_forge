from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def tensor_density_matrix(circuit: cirq.Circuit, qubits: Optional[List[cirq.Qid]]=None) -> np.ndarray:
    """Given a circuit with mixtures or channels, contract a tensor network
    representing the resultant density matrix.

    Note: If the circuit contains 6 qubits or fewer, we use a bespoke
    contraction ordering that corresponds to the "normal" in-time contraction
    ordering. Otherwise, the contraction order determination could take
    longer than doing the contraction. Your mileage may vary and benchmarking
    is encouraged for your particular problem if performance is important.
    """
    if qubits is None:
        qubits = sorted(circuit.all_qubits())
    tensors, qubit_frontier, _ = circuit_to_density_matrix_tensors(circuit=circuit, qubits=qubits)
    tn = qtn.TensorNetwork(tensors)
    f_inds = tuple((f'nf{qubit_frontier[q]}_q{q}' for q in qubits))
    b_inds = tuple((f'nb{qubit_frontier[q]}_q{q}' for q in qubits))
    if len(qubits) <= 6:
        tags_seq = [(f'i{i}b', f'i{i}f') for i in range(len(circuit) + 1)]
        tn.contract_cumulative(tags_seq, inplace=True)
    else:
        tn.contract(inplace=True)
    return tn.to_dense(f_inds, b_inds)