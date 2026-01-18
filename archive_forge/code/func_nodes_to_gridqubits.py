import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def nodes_to_gridqubits(self, offset=(0, 0)) -> Dict[Tuple[int, int], 'cirq.GridQubit']:
    """Return a mapping from graph nodes to `cirq.GridQubit`

        Args:
            offset: Offset row and column indices of the resultant GridQubits by this amount.
                The offset positions the top-left node in the `draw(tilted=False)` frame.
        """
    return {(r, c): GridQubit(r, c) + offset for r, c in self.graph.nodes}