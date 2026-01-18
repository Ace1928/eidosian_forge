import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
@transform
def pattern_matching_optimization(tape: QuantumTape, pattern_tapes, custom_quantum_cost=None) -> (Sequence[QuantumTape], Callable):
    """Quantum function transform to optimize a circuit given a list of patterns (templates).

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit to be optimized.
        pattern_tapes(list(.QuantumTape)): List of quantum tapes that implement the identity.
        custom_quantum_cost (dict): Optional, quantum cost that overrides the default cost dictionary.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        QuantumFunctionError: The pattern provided is not a valid QuantumTape or the pattern contains measurements or
            the pattern does not implement identity or the circuit has less qubits than the pattern.

    **Example**

    >>> dev = qml.device('default.qubit', wires=5)

    You can apply the transform directly on a :class:`QNode`. For that, you need first to define a pattern that is to be
    found in the circuit. We use the following pattern that implements the identity:

    .. code-block:: python

        ops = [qml.S(0), qml.S(0), qml.Z(0)]
        pattern = qml.tape.QuantumTape(ops)


    Let's consider the following circuit where we want to replace a sequence of two ``pennylane.S`` gates with a
    ``pennylane.PauliZ`` gate.

    .. code-block:: python

        @partial(pattern_matching_optimization, pattern_tapes = [pattern])
        @qml.qnode(device=dev)
        def circuit():
            qml.S(wires=0)
            qml.Z(0)
            qml.S(wires=1)
            qml.CZ(wires=[0, 1])
            qml.S(wires=1)
            qml.S(wires=2)
            qml.CZ(wires=[1, 2])
            qml.S(wires=2)
            return qml.expval(qml.X(0))

    During the call of the circuit, it is first optimized (if possible) and then executed.

    >>> circuit()

    .. details::
        :title: Usage Details

        .. code-block:: python

            def circuit():
                qml.S(wires=0)
                qml.Z(0)
                qml.S(wires=1)
                qml.CZ(wires=[0, 1])
                qml.S(wires=1)
                qml.S(wires=2)
                qml.CZ(wires=[1, 2])
                qml.S(wires=2)
                return qml.expval(qml.X(0))

        For optimizing the circuit given the following template of CNOTs we apply the ``pattern_matching``
        transform.

        >>> qnode = qml.QNode(circuit, dev)
        >>> optimized_qfunc = pattern_matching_optimization(pattern_tapes=[pattern])(circuit)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        >>> print(qml.draw(qnode)())
        0: ──S──Z─╭●──────────┤  <X>
        1: ──S────╰Z──S─╭●────┤
        2: ──S──────────╰Z──S─┤

        >>> print(qml.draw(optimized_qnode)())
        0: ──S†─╭●────┤  <X>
        1: ──Z──╰Z─╭●─┤
        2: ──Z─────╰Z─┤

        Note that with this pattern we also replace a ``pennylane.S``, ``pennylane.PauliZ`` sequence by
        ``Adjoint(S)``. If one would like avoiding this, it possible to give a custom
        quantum cost dictionary with a negative cost for ``pennylane.PauliZ``.

        >>> my_cost = {"PauliZ": -1 , "S": 1, "Adjoint(S)": 1}
        >>> optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[pattern], custom_quantum_cost=my_cost)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        >>> print(qml.draw(optimized_qnode)())
        0: ──S──Z─╭●────┤  <X>
        1: ──Z────╰Z─╭●─┤
        2: ──Z───────╰Z─┤

        Now we can consider a more complicated example with the following quantum circuit to be optimized

        .. code-block:: python

            def circuit():
                qml.Toffoli(wires=[3, 4, 0])
                qml.CNOT(wires=[1, 4])
                qml.CNOT(wires=[2, 1])
                qml.Hadamard(wires=3)
                qml.Z(1)
                qml.CNOT(wires=[2, 3])
                qml.Toffoli(wires=[2, 3, 0])
                qml.CNOT(wires=[1, 4])
                return qml.expval(qml.X(0))

        We define a pattern that implement the identity:

        .. code-block:: python

            ops = [
                qml.CNOT(wires=[1, 2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[0, 2]),
            ]
            tape = qml.tape.QuantumTape(ops)

        For optimizing the circuit given the given following pattern of CNOTs we apply the `pattern_matching`
        transform.

        >>> dev = qml.device('default.qubit', wires=5)
        >>> qnode = qml.QNode(circuit, dev)
        >>> optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[pattern])
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        In our case, it is possible to find three CNOTs and replace this pattern with only two CNOTs and therefore
        optimizing the circuit. The number of CNOTs in the circuit is reduced by one.

        >>> qml.specs(qnode)()["resources"].gate_types["CNOT"]
        4

        >>> qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]
        3

        >>> print(qml.draw(qnode)())
        0: ─╭X──────────╭X────┤  <X>
        1: ─│──╭●─╭X──Z─│──╭●─┤
        2: ─│──│──╰●─╭●─├●─│──┤
        3: ─├●─│───H─╰X─╰●─│──┤
        4: ─╰●─╰X──────────╰X─┤

        >>> print(qml.draw(optimized_qnode)())
        0: ─╭X──────────╭X─┤  <X>
        1: ─│─────╭X──Z─│──┤
        2: ─│──╭●─╰●─╭●─├●─┤
        3: ─├●─│───H─╰X─╰●─┤
        4: ─╰●─╰X──────────┤

    .. seealso:: :func:`~.pattern_matching`

    **References**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2022.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_
    """
    consecutive_wires = Wires(range(len(tape.wires)))
    inverse_wires_map = OrderedDict(zip(consecutive_wires, tape.wires))
    for pattern in pattern_tapes:
        if not isinstance(pattern, QuantumScript):
            raise qml.QuantumFunctionError('The pattern is not a valid quantum tape.')
        if pattern.measurements:
            raise qml.QuantumFunctionError('The pattern contains measurements.')
        if not np.allclose(qml.matrix(pattern, wire_order=pattern.wires), np.eye(2 ** pattern.num_wires)):
            raise qml.QuantumFunctionError('Pattern is not valid, it does not implement identity.')
        if tape.num_wires < pattern.num_wires:
            raise qml.QuantumFunctionError('Circuit has less qubits than the pattern.')
        circuit_dag = commutation_dag(tape)
        pattern_dag = commutation_dag(pattern)
        max_matches = pattern_matching(circuit_dag, pattern_dag)
        if max_matches:
            substitution = TemplateSubstitution(max_matches, circuit_dag, pattern_dag, custom_quantum_cost)
            substitution.substitution()
            already_sub = []
            if substitution.substitution_list:
                with qml.queuing.AnnotatedQueue() as q_inside:
                    for group in substitution.substitution_list:
                        circuit_sub = group.circuit_config
                        template_inverse = group.template_config
                        pred = group.pred_block
                        qubit = group.qubit_config[0]
                        for elem in pred:
                            node = circuit_dag.get_node(elem)
                            inst = copy.deepcopy(node.op)
                            qml.apply(inst)
                            already_sub.append(elem)
                        already_sub = already_sub + circuit_sub
                        for index in template_inverse:
                            all_qubits = tape.wires.tolist()
                            all_qubits.sort()
                            wires_t = group.template_dag.get_node(index).wires
                            wires_c = [qubit[x] for x in wires_t]
                            wires = [all_qubits[x] for x in wires_c]
                            node = group.template_dag.get_node(index)
                            inst = copy.deepcopy(node.op)
                            inst = qml.map_wires(inst, wire_map=dict(zip(inst.wires, wires)))
                            adjoint(qml.apply, lazy=False)(inst)
                    for node_id in substitution.unmatched_list:
                        node = circuit_dag.get_node(node_id)
                        inst = copy.deepcopy(node.op)
                        qml.apply(inst)
                qscript = QuantumScript.from_queue(q_inside)
                [tape], _ = qml.map_wires(input=qscript, wire_map=inverse_wires_map)
    new_tape = type(tape)(tape.operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([new_tape], null_postprocessing)