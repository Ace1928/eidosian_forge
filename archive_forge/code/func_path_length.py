import warnings
import networkx as nx
def path_length(v):
    if method == 'unweighted':
        return nx.single_source_shortest_path_length(G, v)
    elif method == 'dijkstra':
        return nx.single_source_dijkstra_path_length(G, v, weight=weight)
    elif method == 'bellman-ford':
        return nx.single_source_bellman_ford_path_length(G, v, weight=weight)