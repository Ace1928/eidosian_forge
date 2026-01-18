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
def swap_along_path(self, path: Tuple[ops.Qid]):
    """Adds SWAPs to move a logical qubit along a specified path."""
    for i in range(len(path) - 1):
        self.apply_swap(cast(QidPair, path[i:i + 2]))