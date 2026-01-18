import pytest
import networkx as nx
def build_compressed_graph(self):
    compressed_matrix = [('1', ['B', 'C']), ('2', ['ABC']), ('3', ['6AB']), ('4', ['ABC']), ('5', ['6AB']), ('6', ['6AB', 'A']), ('A', ['6AB', 'ABC']), ('B', ['ABC', '6AB']), ('C', ['ABC'])]
    compressed_graph = nx.Graph()
    for source, targets in compressed_matrix:
        for target in targets:
            compressed_graph.add_edge(source, target)
    return compressed_graph