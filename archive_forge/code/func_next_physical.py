from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
def next_physical(current_physical: 'cirq.Qid', partner: 'cirq.Qid', isolated: bool=False) -> 'cirq.Qid':
    if current_physical not in mapped_physicals:
        return current_physical
    if not isolated:
        sorted_neighbors = sorted(self.device_graph.neighbors(current_physical), key=lambda x: self.device_graph.degree(x), reverse=True)
        for neighbor in sorted_neighbors:
            if neighbor not in mapped_physicals:
                return neighbor
    return self._closest_unmapped_qubit(partner, mapped_physicals)