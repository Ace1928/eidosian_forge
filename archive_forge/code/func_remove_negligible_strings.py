import networkx
from cirq import circuits, linalg
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import pauli_string_dag_from_circuit
from cirq.contrib.paulistring.recombine import move_pauli_strings_into_circuit
from cirq.contrib.paulistring.separate import convert_and_separate_circuit
from cirq.ops import PauliStringGateOperation
def remove_negligible_strings(string_dag: circuitdag.CircuitDag, atol=1e-08) -> None:
    for node in tuple(string_dag.nodes()):
        if linalg.all_near_zero_mod(node.val.exponent_relative, 2, atol=atol):
            string_dag.remove_node(node)