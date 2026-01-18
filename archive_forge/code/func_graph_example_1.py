import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def graph_example_1():
    G = nx.convert_node_labels_to_integers(nx.grid_graph([5, 5]), label_attribute='labels')
    rlabels = nx.get_node_attributes(G, 'labels')
    labels = {v: k for k, v in rlabels.items()}
    for nodes in [(labels[0, 0], labels[1, 0]), (labels[0, 4], labels[1, 4]), (labels[3, 0], labels[4, 0]), (labels[3, 4], labels[4, 4])]:
        new_node = G.order() + 1
        P = nx.petersen_graph()
        G = nx.disjoint_union(G, P)
        G.add_edge(new_node + 1, nodes[0])
        G.add_edge(new_node, nodes[1])
        K = nx.complete_graph(5)
        G = nx.disjoint_union(G, K)
        G.add_edge(new_node + 2, new_node + 11)
        G.add_edge(new_node + 3, new_node + 12)
        G.add_edge(new_node + 4, new_node + 13)
        G = nx.disjoint_union(G, K)
        nbrs = G[new_node + 10]
        G.remove_node(new_node + 10)
        for nbr in nbrs:
            G.add_edge(new_node + 17, nbr)
        G.add_edge(new_node + 16, new_node + 5)
    return G