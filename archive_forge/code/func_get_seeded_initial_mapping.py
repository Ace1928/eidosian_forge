import random
import numpy as np
import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def get_seeded_initial_mapping(graph_seed, init_seed):
    logical_graph = nx.erdos_renyi_graph(10, 0.5, seed=graph_seed)
    logical_graph = nx.relabel_nodes(logical_graph, cirq.LineQubit)
    device_graph = ccr.get_grid_device_graph(4, 4)
    return ccr.initialization.get_initial_mapping(logical_graph, device_graph, init_seed)