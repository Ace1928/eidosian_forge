import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
@classmethod
def from_heavy_square(cls, distance, bidirectional=True) -> 'CouplingMap':
    """Return a heavy square graph coupling map.

        A heavy square graph is described in:

        https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011022

        Args:
            distance (int): The code distance for the generated heavy square
                graph. The value for distance can be any odd positive integer.
                The distance relates to the number of qubits by:
                :math:`n = 3d^2 - 2d` where :math:`n` is the
                number of qubits and :math:`d` is the ``distance`` parameter.
            bidirectional (bool): Whether the edges in the output coupling
                graph are bidirectional or not. By default this is set to
                ``True``
        Returns:
            CouplingMap: A heavy square coupling graph
        """
    cmap = cls(description='heavy-square')
    cmap.graph = rx.generators.directed_heavy_square_graph(distance, bidirectional=bidirectional)
    return cmap