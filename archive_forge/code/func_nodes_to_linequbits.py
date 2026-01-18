import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def nodes_to_linequbits(self, offset: int=0) -> Dict[int, 'cirq.LineQubit']:
    """Return a mapping from graph nodes to `cirq.LineQubit`

        Args:
            offset: Offset integer positions of the resultant LineQubits by this amount.
        """
    return dict(enumerate(LineQubit.range(offset, offset + self.n_nodes)))