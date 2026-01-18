import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def generalized_singles(wires, delta_sz):
    """Return generalized single excitation terms

    .. math::
        \\hat{T_1} = \\sum_{pq} t_{p}^{q} \\hat{c}^{\\dagger}_{q} \\hat{c}_{p}

    """
    sz = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(len(wires))])
    gen_singles_wires = []
    for r in range(len(wires)):
        for p in range(len(wires)):
            if sz[p] - sz[r] == delta_sz and p != r:
                if r < p:
                    gen_singles_wires.append(wires[r:p + 1])
                else:
                    gen_singles_wires.append(wires[p:r + 1][::-1])
    return gen_singles_wires