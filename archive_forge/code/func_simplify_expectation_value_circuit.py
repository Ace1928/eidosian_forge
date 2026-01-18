from typing import Iterator
import networkx as nx
import cirq
def simplify_expectation_value_circuit(circuit_sand: cirq.Circuit):
    """For low weight operators on low-degree circuits, we can simplify
    the circuit representation of an expectation value.

    In particular, this should be used on `circuit_for_expectation_value`
    circuits. It will merge single- and two-qubit gates from the "forwards"
    and "backwards" parts of the circuit outside of the operator's lightcone.

    This might be too slow in practice and you can just use quimb to simplify
    things for you.
    """
    n_op = sum((1 for _ in circuit_sand.all_operations()))
    circuit = circuit_sand.copy()
    while True:
        circuit = cirq.merge_k_qubit_unitaries(circuit, k=1)
        circuit = cirq.drop_negligible_operations(circuit, atol=1e-06)
        circuit = cirq.merge_k_qubit_unitaries(circuit, k=2)
        circuit = cirq.drop_empty_moments(circuit)
        new_n_op = sum((1 for _ in circuit.all_operations()))
        if new_n_op >= n_op:
            break
        n_op = new_n_op
    circuit_sand._moments = circuit._moments