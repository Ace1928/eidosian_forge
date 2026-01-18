import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def gray_code(rank):
    """Generates the Gray code of given rank.

    Args:
        rank (int): rank of the Gray code (i.e. number of bits)
    """

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return
        for i in range(k - 1, -1, -1):
            char = '1' + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = '0' + g[i]
        gray_code_recurse(g, rank - 1)
    g = ['0', '1']
    gray_code_recurse(g, rank - 1)
    return g