from typing import List
from numpy import sqrt
import numpy as np
import cirq
@staticmethod
def parallelep(rows: int, cols: int, lays: int, x0: float=0, y0: float=0, z0: float=0) -> List['ThreeDQubit']:
    """Returns a parallelepiped of ThreeDQubits.

        Args:
            rows: Number of rows in the parallelepiped.
            cols: Number of columns in the parallelepiped.
            lays: Number of layers in the parallelepiped.
            x0: x-coordinate of the first qubit.
            y0: y-coordinate of the first qubit.
            z0: z-coordinate of the first qubit.

        Returns:
            A list of ThreeDQubits filling in a 3d grid
        """
    return [ThreeDQubit(x0 + x, y0 + y, z0 + z) for z in range(lays) for y in range(cols) for x in range(rows)]