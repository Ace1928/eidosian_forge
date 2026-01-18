import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def check_embedding(G, embedding):
    """Raises an exception if the combinatorial embedding is not correct

    Parameters
    ----------
    G : NetworkX graph
    embedding : a dict mapping nodes to a list of edges
        This specifies the ordering of the outgoing edges from a node for
        a combinatorial embedding

    Notes
    -----
    Checks the following things:
        - The type of the embedding is correct
        - The nodes and edges match the original graph
        - Every half edge has its matching opposite half edge
        - No intersections of edges (checked by Euler's formula)
    """
    if not isinstance(embedding, nx.PlanarEmbedding):
        raise nx.NetworkXException('Bad embedding. Not of type nx.PlanarEmbedding')
    embedding.check_structure()
    assert set(G.nodes) == set(embedding.nodes), "Bad embedding. Nodes don't match the original graph."
    g_edges = set()
    for edge in G.edges:
        if edge[0] != edge[1]:
            g_edges.add((edge[0], edge[1]))
            g_edges.add((edge[1], edge[0]))
    assert g_edges == set(embedding.edges), "Bad embedding. Edges don't match the original graph."