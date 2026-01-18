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
def phys_to_log(self, *qubits: 'cirq.Qid') -> Iterable[Optional[ops.Qid]]:
    """Returns an iterator over the logical qubits that map to the given
        physical qubits."""
    return (self._phys_to_log[q] for q in qubits)