import pytest
import networkx as nx
def build_summary_graph(self):
    nodes = {'Supernode-0': {'color': 'Red'}, 'Supernode-1': {'color': 'Blue'}, 'Supernode-2': {'color': 'Yellow'}, 'Supernode-3': {'color': 'Blue'}}
    edges = [('Supernode-0', 'Supernode-1', ['Weak', 'Strong']), ('Supernode-0', 'Supernode-2', ['Weak', 'Strong']), ('Supernode-1', 'Supernode-2', ['Strong']), ('Supernode-1', 'Supernode-3', ['Weak', 'Strong']), ('Supernode-3', 'Supernode-2', ['Strong'])]
    G = nx.MultiDiGraph()
    for node in nodes:
        attributes = nodes[node]
        G.add_node(node, **attributes)
    for source, target, types in edges:
        for type in types:
            G.add_edge(source, target, type=type)
    supernodes = {'Supernode-0': {'A', 'B'}, 'Supernode-1': {'C', 'D'}, 'Supernode-2': {'E', 'F'}, 'Supernode-3': {'G', 'H'}}
    nx.set_node_attributes(G, supernodes, 'group')
    return G