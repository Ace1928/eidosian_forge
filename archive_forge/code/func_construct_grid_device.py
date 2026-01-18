from typing import TYPE_CHECKING
import networkx as nx
from cirq import devices, ops
def construct_grid_device(m: int, n: int) -> RoutingTestingDevice:
    return RoutingTestingDevice(nx.grid_2d_graph(m, n))