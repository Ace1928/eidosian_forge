import warnings
from typing import Sequence, Callable
import numpy as np
from scipy.sparse.linalg import expm
import pennylane as qml
from pennylane import transform
from pennylane.tape import QuantumTape
from pennylane.queuing import QueuingManager
def algebra_commutator(tape, observables, lie_algebra_basis_names, nqubits):
    """Calculate the Riemannian gradient in the Lie algebra with the parameter shift rule
    (see :meth:`RiemannianGradientOptimizer.get_omegas`).

    Args:
        tape (.QuantumTape or .QNode): input circuit
        observables (list[.Observable]): list of observables to be measured. Can be grouped.
        lie_algebra_basis_names (list[str]): List of strings corresponding to valid Pauli words.
        nqubits (int): the number of qubits.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Riemannian gradient function
          of the QNode that can be executed with the same arguments as the QNode to obtain
          the Lie algebra commutator.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Lie algebra commutator.
    """
    tapes_plus_total = []
    tapes_min_total = []
    for obs in observables:
        for o in obs:
            queues_plus = [qml.queuing.AnnotatedQueue() for _ in lie_algebra_basis_names]
            queues_min = [qml.queuing.AnnotatedQueue() for _ in lie_algebra_basis_names]
            for op in tape.operations:
                for t in queues_plus + queues_min:
                    with t:
                        qml.apply(op)
            for i, t in enumerate(queues_plus):
                with t:
                    qml.PauliRot(np.pi / 2, lie_algebra_basis_names[i], wires=list(range(nqubits)))
                    qml.expval(o)
            for i, t in enumerate(queues_min):
                with t:
                    qml.PauliRot(-np.pi / 2, lie_algebra_basis_names[i], wires=list(range(nqubits)))
                    qml.expval(o)
            tapes_plus_total.extend([qml.tape.QuantumScript(*qml.queuing.process_queue(q)) for q, p in zip(queues_plus, lie_algebra_basis_names)])
            tapes_min_total.extend([qml.tape.QuantumScript(*qml.queuing.process_queue(q)) for q, p in zip(queues_min, lie_algebra_basis_names)])
    return tapes_plus_total + tapes_min_total