import networkx
from cirq import circuits, linalg
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import pauli_string_dag_from_circuit
from cirq.contrib.paulistring.recombine import move_pauli_strings_into_circuit
from cirq.contrib.paulistring.separate import convert_and_separate_circuit
from cirq.ops import PauliStringGateOperation
def merge_equal_strings(string_dag: circuitdag.CircuitDag) -> None:
    for node in tuple(string_dag.nodes()):
        if node not in string_dag.nodes():
            continue
        commuting_nodes = set(string_dag.nodes()) - set(networkx.dag.ancestors(string_dag, node)) - set(networkx.dag.descendants(string_dag, node)) - set([node])
        for other_node in commuting_nodes:
            if node.val.pauli_string.equal_up_to_coefficient(other_node.val.pauli_string):
                string_dag.remove_node(other_node)
                node.val = node.val.merged_with(other_node.val)