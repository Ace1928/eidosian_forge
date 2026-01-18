import operator
from typing import Callable, Iterable, List, TYPE_CHECKING
import re
import networkx as nx
from cirq import circuits, ops
import cirq.contrib.acquaintance as cca
from cirq.contrib.circuitdag import CircuitDag
from cirq.contrib.routing.swap_network import SwapNetwork
def is_valid_routing(circuit: circuits.Circuit, swap_network: SwapNetwork, *, equals: BINARY_OP_PREDICATE=operator.eq, can_reorder: BINARY_OP_PREDICATE=lambda op1, op2: not set(op1.qubits) & set(op2.qubits)) -> bool:
    """Determines whether a swap network is consistent with a given circuit.

    Args:
        circuit: The circuit.
        swap_network: The swap network whose validity is to be checked.
        equals: The function to determine equality of operations. Defaults to
            `operator.eq`.
        can_reorder: A predicate that determines if two operations may be
            reordered.

    Raises:
        ValueError: If equals operator or can_reorder throws a ValueError.
    """
    circuit_dag = CircuitDag.from_circuit(circuit, can_reorder=can_reorder)
    logical_operations = swap_network.get_logical_operations()
    try:
        return cca.is_topologically_sorted(circuit_dag, logical_operations, equals)
    except ValueError as err:
        if re.match('Operation .* acts on unmapped qubit .*\\.', str(err)):
            return False
        raise