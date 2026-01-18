from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def nodes_between(self, a, b):
    """Nodes on all the directed paths between the two given nodes.

        Returns the set of all nodes ``s`` that fulfill :math:`a \\le s \\le b`.
        There is a directed path from ``a`` via ``s`` to ``b`` iff the set is nonempty.
        The endpoints belong to the path.

        Args:
            a (Operator): initial node
            b (Operator): final node

        Returns:
            list[Operator]: nodes on all the directed paths between a and b
        """
    A = self.descendants([a])
    A.append(a)
    B = self.ancestors([b])
    B.append(b)
    return [B.pop(i) for op1 in A for i, op2 in enumerate(B) if op1 is op2]