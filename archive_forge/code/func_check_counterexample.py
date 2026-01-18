import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def check_counterexample(G, sub_graph):
    """Raises an exception if the counterexample is wrong.

    Parameters
    ----------
    G : NetworkX graph
    subdivision_nodes : set
        A set of nodes inducing a subgraph as a counterexample
    """
    sub_graph = nx.Graph(sub_graph)
    for u in sub_graph:
        if sub_graph.has_edge(u, u):
            sub_graph.remove_edge(u, u)
    contract = list(sub_graph)
    while len(contract) > 0:
        contract_node = contract.pop()
        if contract_node not in sub_graph:
            continue
        degree = sub_graph.degree[contract_node]
        if degree == 2:
            neighbors = iter(sub_graph[contract_node])
            u = next(neighbors)
            v = next(neighbors)
            contract.append(u)
            contract.append(v)
            sub_graph.remove_node(contract_node)
            sub_graph.add_edge(u, v)
    if len(sub_graph) == 5:
        if not nx.is_isomorphic(nx.complete_graph(5), sub_graph):
            raise nx.NetworkXException('Bad counter example.')
    elif len(sub_graph) == 6:
        if not nx.is_isomorphic(nx.complete_bipartite_graph(3, 3), sub_graph):
            raise nx.NetworkXException('Bad counter example.')
    else:
        raise nx.NetworkXException('Bad counter example.')