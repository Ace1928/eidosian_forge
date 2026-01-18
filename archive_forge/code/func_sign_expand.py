import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
@transform
def sign_expand(tape: qml.tape.QuantumTape, circuit=False, J=10, delta=0.0, controls=('Hadamard', 'Target')) -> (Sequence[qml.tape.QuantumTape], Callable):
    """
    Splits a tape measuring a (fast-forwardable) Hamiltonian expectation into mutliple tapes of
    the Xi or sgn decomposition, and provides a function to recombine the results.

    Implementation of ideas from arXiv:2207.09479

    For the calculation of variances, one assumes an even distribution of shots among the groups.

    Args:
        tape (QNode or QuantumTape): the quantum circuit used when calculating the expectation value of the Hamiltonian
        circuit (bool): Toggle the calculation of the analytical Xi decomposition or if True
          constructs the circuits of the approximate sign decomposition to measure the expectation
          value
        J (int): The times the time evolution of the hamiltonian is repeated in the quantum signal
          processing approximation of the sgn-decomposition
        delta (float): The minimal
        controls (List[control1, control2]): The two additional controls to implement the
          Hadamard test and the quantum signal processing part on, have to be wires on the device

    Returns:
        qnode (pennylane.QNode) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.Z(0) + 0.5 * qml.Z(2) + qml.Z(1)

    a device with auxiliary qubits,

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])

    and a circuit of the form, with the transform as decorator.

    .. code-block:: python3

        @qml.transforms.sign_expand
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.X(2)
            return qml.expval(H)

    >>> circuit()
    -0.4999999999999999

    You can also work directly on tapes:

    .. code-block:: python3

            operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), qml.X(2)]
            measurements = [qml.expval(H)]
            tape = qml.tape.QuantumTape(operations, measurements)

    We can use the ``sign_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian in these new decompositions

    >>> tapes, fn = qml.transforms.sign_expand(tape)

    We can evaluate these tapes on a device, it needs two additional ancilla gates labeled 'Hadamard' and 'Target' if
    one wants to make the circuit approximation of the decomposition:

    >>> dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
    >>> res = dev.execute(tapes)
    >>> fn(res)
    -0.4999999999999999

    To evaluate the circuit approximation of the decomposition one can construct the sgn-decomposition by changing the
    kwarg circuit to True:

    >>> tapes, fn = qml.transforms.sign_expand(tape, circuit=True, J=20, delta=0)
    >>> dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
    >>> dev.execute(tapes)
    >>> fn(res)
    -0.24999999999999994


    Lastly, as the paper is about minimizing variance, one can also calculate the variance of the estimator by
    changing the tape:


    .. code-block:: python3

            operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), qml.X(2)]
            measurements = [qml.var(H)]
            tape = qml.tape.QuantumTape(operations, measurements)

    >>> tapes, fn = qml.transforms.sign_expand(tape, circuit=True, J=20, delta=0)
    >>> dev = qml.device("default.qubit", wires=[0,1,2,'Hadamard','Target'])
    >>> res = dev.execute(tapes)
    >>> fn(res)
    10.108949481425782

    """
    path_str = path.dirname(__file__)
    with open(path_str + '/sign_expand_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    phis = list(filter(lambda data: data['delta'] == delta and data['order'] == J, data))[0]['opt_params']
    hamiltonian = tape.measurements[0].obs
    wires = hamiltonian.wires
    if not isinstance(hamiltonian, qml.Hamiltonian) or len(tape.measurements) > 1 or tape.measurements[0].return_type not in [qml.measurements.Expectation, qml.measurements.Variance]:
        raise ValueError("Passed tape must end in `qml.expval(H)` or 'qml.var(H)', where H is of type `qml.Hamiltonian`")
    hamiltonian.compute_grouping()
    if len(hamiltonian.grouping_indices) != 1:
        raise ValueError('Passed hamiltonian must be jointly measurable')
    dEs, mus, times, projs = calculate_xi_decomposition(hamiltonian)
    if circuit:
        tapes = construct_sgn_circuit(hamiltonian, tape, mus, times, phis, controls)
        if tape.measurements[0].return_type == qml.measurements.Expectation:

            def processing_fn(res):
                products = [a * b for a, b in zip(res, dEs)]
                return qml.math.sum(products)
        else:

            def processing_fn(res):
                products = [a * b for a, b in zip(res, dEs)]
                return qml.math.sum(products) * len(products)
        return (tapes, processing_fn)
    tapes = []
    for proj in projs:
        if tape.measurements[0].return_type == qml.measurements.Expectation:
            measurements = [qml.expval(qml.Hermitian(proj, wires=wires))]
        else:
            measurements = [qml.var(qml.Hermitian(proj, wires=wires))]
        new_tape = qml.tape.QuantumScript(tape.operations, measurements, shots=tape.shots)
        tapes.append(new_tape)

    def processing_fn(res):
        return qml.math.sum(res) if tape.measurements[0].return_type == qml.measurements.Expectation else qml.math.sum(res) * len(res)
    return (tapes, processing_fn)