from typing import List, Sequence, Callable
import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess
from pennylane.ops import SProd, Sum
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform
@transform
def hamiltonian_expand(tape: QuantumTape, group: bool=True) -> (Sequence[QuantumTape], Callable):
    """
    Splits a tape measuring a Hamiltonian expectation into mutliple tapes of Pauli expectations,
    and provides a function to recombine the results.

    Args:
        tape (QNode or QuantumTape or Callable): the quantum circuit used when calculating the
            expectation value of the Hamiltonian
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.Y(2) @ qml.Z(1) + 0.5 * qml.Z(2) + qml.Z(1)

    and a tape of the form,

    .. code-block:: python3

        ops = [qml.Hadamard(0), qml.CNOT((0,1)), qml.X(2)]
        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    We can use the ``hamiltonian_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    array(-0.5)

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.Z(0), qml.X(1), qml.X(0)])

        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    With grouping, the Hamiltonian gets split into two groups of observables (here ``[qml.Z(0)]`` and
    ``[qml.X(1), qml.X(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.Z(0)]``, ``[qml.X(1)]`` and ``[qml.X(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    3

    Alternatively, if the Hamiltonian has already computed groups, they are used even if ``group=False``:

    .. code-block:: python3

        obs = [qml.Z(0), qml.X(1), qml.X(0)]
        coeffs = [1., 2., 3.]
        H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')

        # the initialisation already computes grouping information and stores it in the Hamiltonian
        assert H.grouping_indices is not None

        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    Grouping information has been used to reduce the number of tapes from 3 to 2:

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    2
    """
    if len(tape.measurements) != 1 or not isinstance((hamiltonian := tape.measurements[0].obs), qml.Hamiltonian) or (not isinstance(tape.measurements[0], ExpectationMP)):
        raise ValueError('Passed tape must end in `qml.expval(H)`, where H is of type `qml.Hamiltonian`')
    if qml.math.shape(hamiltonian.coeffs) == (0,) and qml.math.shape(hamiltonian.ops) == (0,):
        raise ValueError('The Hamiltonian in the tape has no terms defined - cannot perform the Hamiltonian expansion.')
    if group or hamiltonian.grouping_indices is not None:
        if hamiltonian.grouping_indices is None:
            hamiltonian.compute_grouping()
        coeff_groupings = [qml.math.stack([hamiltonian.data[i] for i in indices]) for indices in hamiltonian.grouping_indices]
        obs_groupings = [[hamiltonian.ops[i] for i in indices] for indices in hamiltonian.grouping_indices]
        tapes = []
        for obs in obs_groupings:
            new_tape = tape.__class__(tape.operations, (qml.expval(o) for o in obs), shots=tape.shots)
            new_tape = new_tape.expand(stop_at=lambda obj: True)
            tapes.append(new_tape)

        def processing_fn(res_groupings):
            res_groupings = [qml.math.stack(r) if isinstance(r, (tuple, qml.numpy.builtins.SequenceBox)) else r for r in res_groupings]
            res_groupings = [qml.math.reshape(r, (1,)) if r.shape == () else r for r in res_groupings]
            dot_products = []
            for c_group, r_group in zip(coeff_groupings, res_groupings):
                if tape.batch_size:
                    r_group = r_group.T
                if len(c_group) == 1 and len(r_group) != 1:
                    dot_products.append(r_group * c_group)
                else:
                    dot_products.append(qml.math.dot(r_group, c_group))
            summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
            return qml.math.convert_like(summed_dot_products, res_groupings[0])
        return (tapes, processing_fn)
    coeffs = hamiltonian.data
    tapes = []
    for o in hamiltonian.ops:
        new_tape = tape.__class__(tape.operations, [qml.expval(o)], shots=tape.shots)
        tapes.append(new_tape)

    def processing_fn(res):
        dot_products = []
        for c, r in zip(coeffs, res):
            if qml.math.ndim(c) == 0 and qml.math.size(r) != 1:
                dot_products.append(qml.math.squeeze(r) * c)
            else:
                dot_products.append(qml.math.dot(qml.math.squeeze(r), c))
        summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
        return qml.math.convert_like(summed_dot_products, res[0])
    return (tapes, processing_fn)