from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
    """Maps disjoint lines of logical qubits onto lines of physical qubits.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            a dictionary that maps logical qubits in the circuit (keys) to physical qubits on the
            device (values).
        """
    mapped_physicals: Set['cirq.Qid'] = set()
    qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = {}
    circuit_graph, partners = self._make_circuit_graph(circuit)

    def next_physical(current_physical: 'cirq.Qid', partner: 'cirq.Qid', isolated: bool=False) -> 'cirq.Qid':
        if current_physical not in mapped_physicals:
            return current_physical
        if not isolated:
            sorted_neighbors = sorted(self.device_graph.neighbors(current_physical), key=lambda x: self.device_graph.degree(x), reverse=True)
            for neighbor in sorted_neighbors:
                if neighbor not in mapped_physicals:
                    return neighbor
        return self._closest_unmapped_qubit(partner, mapped_physicals)
    pq = self.center
    for logical_line in circuit_graph:
        for lq in logical_line:
            is_isolated = len(logical_line) == 1
            partner = qubit_map[partners[lq]] if lq in partners and is_isolated else self.center
            pq = next_physical(pq, partner, isolated=is_isolated)
            mapped_physicals.add(pq)
            qubit_map[lq] = pq
    return qubit_map