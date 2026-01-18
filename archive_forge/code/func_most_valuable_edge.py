import networkx as nx
def most_valuable_edge(G):
    """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
    betweenness = nx.edge_betweenness_centrality(G)
    return max(betweenness, key=betweenness.get)