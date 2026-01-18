import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def nodes_as_gridqubits(self) -> List['cirq.GridQubit']:
    """Get the graph nodes as cirq.GridQubit"""
    return [GridQubit(r, c) for r, c in sorted(self.graph.nodes)]