from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx
import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset
def minimal_distance(self) -> float:
    """Returns the minimal distance between two qubits in qubits.

        Args:
            qubits: qubit involved in the distance computation

        Raises:
            ValueError: If the device has only one qubit

        Returns:
            The minimal distance between qubits, in spacial coordinate units.
        """
    if len(self.qubits) <= 1:
        raise ValueError('Two qubits to compute a minimal distance.')
    return min([self.distance(q1, q2) for q1 in self.qubits for q2 in self.qubits if q1 != q2])