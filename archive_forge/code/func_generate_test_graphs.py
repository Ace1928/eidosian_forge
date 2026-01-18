import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def generate_test_graphs():
    """
    Generate a gauntlet of different test graphs with different properties
    """
    import random
    rng = random.Random(976689776)
    num_randomized = 3
    for directed in [0, 1]:
        cls = nx.DiGraph if directed else nx.Graph
        for num_nodes in range(17):
            graph = cls()
            graph.add_nodes_from(range(num_nodes))
            yield graph
            if num_nodes > 0:
                for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for seed in range(num_randomized):
                        graph = nx.erdos_renyi_graph(num_nodes, p, directed=directed, seed=rng)
                        yield graph
                yield nx.complete_graph(num_nodes, cls)
        yield nx.path_graph(3, create_using=cls)
        yield nx.balanced_tree(r=1, h=3, create_using=cls)
        if not directed:
            yield nx.circular_ladder_graph(4, create_using=cls)
            yield nx.star_graph(5, create_using=cls)
            yield nx.lollipop_graph(4, 2, create_using=cls)
            yield nx.wheel_graph(7, create_using=cls)
            yield nx.dorogovtsev_goltsev_mendes_graph(4, create_using=cls)