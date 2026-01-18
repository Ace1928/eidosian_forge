from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
def degree_lt_two(q: 'cirq.Qid'):
    return any((circuit_graph[component_id[q]][i] == q for i in [-1, 0]))