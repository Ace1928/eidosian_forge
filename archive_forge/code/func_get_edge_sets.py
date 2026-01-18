import itertools
from typing import (
import numpy as np
import networkx as nx
from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib import circuitdag
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph
def get_edge_sets(self, edge_set_size: int) -> Iterable[Sequence[QidPair]]:
    """Returns matchings of the device graph of a given size."""
    if edge_set_size not in self.edge_sets:
        self.edge_sets[edge_set_size] = [cast(Sequence[QidPair], edge_set) for edge_set in itertools.combinations(self.device_graph.edges, edge_set_size) if all((set(e).isdisjoint(f) for e, f in itertools.combinations(edge_set, 2)))]
    return self.edge_sets[edge_set_size]