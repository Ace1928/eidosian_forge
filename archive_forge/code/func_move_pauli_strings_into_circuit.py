from typing import Any, Callable, Iterable, Sequence, Tuple, Union, cast, List
from cirq import circuits, ops, protocols
from cirq.contrib import circuitdag
from cirq.contrib.paulistring.pauli_string_dag import (
def move_pauli_strings_into_circuit(circuit_left: Union[circuits.Circuit, circuitdag.CircuitDag], circuit_right: circuits.Circuit) -> circuits.Circuit:
    if isinstance(circuit_left, circuitdag.CircuitDag):
        string_dag = circuitdag.CircuitDag(pauli_string_reorder_pred, circuit_left)
    else:
        string_dag = pauli_string_dag_from_circuit(cast(circuits.Circuit, circuit_left))
    output_ops = list(circuit_right.all_operations())
    rightmost_nodes = set(string_dag.nodes()) - set((before for before, _ in string_dag.edges()))
    while rightmost_nodes:
        placements = _sorted_best_string_placements(rightmost_nodes, output_ops)
        last_index = len(output_ops)
        for best_string_op, best_index, best_node in placements:
            assert best_index <= last_index, f'Unexpected insertion index order, {best_index} >= {last_index}, len: {len(output_ops)}'
            last_index = best_index
            output_ops.insert(best_index, best_string_op)
            rightmost_nodes.remove(best_node)
            rightmost_nodes.update((pred_node for pred_node in string_dag.predecessors(best_node) if len(string_dag.succ[pred_node]) <= 1))
            string_dag.remove_node(best_node)
    assert not string_dag.nodes(), 'There was a cycle in the CircuitDag'
    return circuits.Circuit(output_ops, strategy=circuits.InsertStrategy.EARLIEST)