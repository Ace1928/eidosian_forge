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
def set_initial_mapping(self, initial_mapping: Optional[Dict[ops.Qid, ops.Qid]]=None):
    """Sets the internal state according to an initial mapping.

        Args:
            initial_mapping: The mapping to use. If not given, one is found
                greedily.
        """
    if initial_mapping is None:
        time_slices = get_time_slices(self.remaining_dag)
        if not time_slices:
            initial_mapping = dict(zip(self.device_graph, self.logical_qubits))
        else:
            logical_graph = time_slices[0]
            logical_graph.add_nodes_from(self.logical_qubits)
            initial_mapping = get_initial_mapping(logical_graph, self.device_graph, self.prng)
    self.initial_mapping = initial_mapping
    self._phys_to_log = {q: initial_mapping.get(q) for q in self.physical_qubits}
    self._log_to_phys = {l: p for p, l in self._phys_to_log.items() if l is not None}
    self._assert_mapping_consistency()