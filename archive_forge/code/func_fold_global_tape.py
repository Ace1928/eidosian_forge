from copy import copy
from typing import Any, Dict, Optional, Sequence, Callable
from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform
import pennylane as qml
def fold_global_tape(circuit, scale_factor):
    """
    This is the internal tape transform to be used with :func:`~.pennylane.transforms.mitigate_with_zne`.
    For the user-facing function see :func:`~.pennylane.transforms.fold_global`.

    Args:
        circuit (QuantumTape): the circuit to be folded
        scale_factor (float): Scale factor :math:`\\lambda` determining :math:`n` and :math:`s`

    Returns:
        QuantumTape: Folded circuit

    """

    def qfunc(op):
        copy(op).queue()
    base_ops = circuit.expand().copy(copy_operations=True).operations
    num_global_folds, fraction_scale = _divmod(scale_factor - 1, 2)
    n_ops = len(base_ops)
    num_to_fold = int(round(fraction_scale * n_ops / 2))
    with AnnotatedQueue() as new_circuit_q:
        for op in base_ops:
            qfunc(op)
        for _ in range(int(num_global_folds)):
            for op in base_ops[::-1]:
                adjoint(qfunc)(op)
            for op in base_ops:
                qfunc(op)
        for i in range(n_ops - 1, n_ops - num_to_fold - 1, -1):
            adjoint(qfunc)(base_ops[i])
        for i in range(n_ops - num_to_fold, n_ops):
            qfunc(base_ops[i])
        for meas in circuit.measurements:
            apply(meas)
    return QuantumScript.from_queue(new_circuit_q)