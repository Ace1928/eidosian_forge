import itertools
from typing import cast, Dict, Hashable, TYPE_CHECKING
import networkx as nx
from sortedcontainers import SortedDict, SortedSet
from cirq import ops, value
def get_initial_mapping(logical_graph: nx.Graph, device_graph: nx.Graph, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> Dict[ops.Qid, ops.Qid]:
    """Gets an initial mapping of logical to physical qubits for routing.

    Args:
        logical_graph: The graph whose edges correspond to pairs of qubits that
            should be mapped to nearby physical qubits.
        device_graph: The graph of the device.
        random_state: Random state or random state seed.

    The mapping starts by mapping the center of the logical graph to the center
    of the physical graph. Subsequent logical qubits are mapped to physical
    qubits greedily. At each iteration, the logical qubits with the largest
    number of already mapped neighbors and the physical qubits neighboring
    those already mapped to are considered. The pair of logical and physical
    qubits that minimizes the average distance to already mapped logical
    neighbors is selected.
    """
    prng = value.parse_random_state(random_state)
    unplaced_vertices = set(logical_graph)
    logical_center = cast(ops.Qid, get_center(logical_graph))
    device_center = cast(ops.Qid, get_center(device_graph))
    mapping = {device_center: logical_center}
    unplaced_vertices.remove(logical_center)
    physical_distances = {(a, b): d for a, neighbor_distances in nx.shortest_path_length(device_graph) for b, d in neighbor_distances.items()}
    while unplaced_vertices:
        placed_vertices = set(mapping.values())
        placed_neighbors = {v: placed_vertices.intersection(logical_graph[v]) for v in unplaced_vertices}
        nums_placed_neighbors = {v: len(N) for v, N in placed_neighbors.items()}
        max_num_placed_neighbors = max(nums_placed_neighbors.values())
        candidates = [v for v, n in nums_placed_neighbors.items() if n == max_num_placed_neighbors]
        border = SortedSet().union(*(device_graph[v] for v in mapping)).difference(mapping)
        total_distances = SortedDict()
        for l, p in itertools.product(candidates, border):
            total_distance = 0
            for pp, ll in mapping.items():
                if logical_graph.has_edge(l, ll):
                    total_distance += physical_distances[p, pp]
            total_distances[l, p] = total_distance
        min_total_distance = min(total_distances.values())
        best_candidates = [lp for lp, d in total_distances.items() if d == min_total_distance]
        choice = prng.choice(len(best_candidates))
        l, p = best_candidates[choice]
        assert p not in mapping
        assert l not in mapping.values()
        mapping[p] = l
        unplaced_vertices.remove(l)
    return mapping