import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def test_heuristic_first_steps(self):
    """Test first steps of min_fill_in heuristic"""
    graph = {n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph}
    print(f'Graph {graph}:')
    elim_node = min_fill_in_heuristic(graph)
    steps = []
    while elim_node is not None:
        print(f'Removing {elim_node}:')
        steps.append(elim_node)
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)
        for u in graph:
            if elim_node in graph[u]:
                graph[u].remove(elim_node)
        del graph[elim_node]
        print(f'Graph {graph}:')
        elim_node = min_fill_in_heuristic(graph)
    assert steps[:2] == [6, 5]