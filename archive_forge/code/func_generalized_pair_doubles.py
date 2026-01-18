import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def generalized_pair_doubles(wires):
    """Return pair coupled-cluster double excitations

    .. math::
        \\hat{T_2} = \\sum_{pq} t_{p_\\alpha p_\\beta}^{q_\\alpha, q_\\beta}
               \\hat{c}^{\\dagger}_{q_\\alpha} \\hat{c}^{\\dagger}_{q_\\beta} \\hat{c}_{p_\\beta} \\hat{c}_{p_\\alpha}

    """
    pair_gen_doubles_wires = [[wires[r:r + 2], wires[p:p + 2]] for r in range(0, len(wires) - 1, 2) for p in range(0, len(wires) - 1, 2) if p != r]
    return pair_gen_doubles_wires