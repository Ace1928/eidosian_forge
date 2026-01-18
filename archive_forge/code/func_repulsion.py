import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def repulsion(*args):
    """Construct the electron repulsion tensor for a given set of basis functions.

        Permutational symmetries are taken from [D.F. Brailsford and G.G. Hall, International
        Journal of Quantum Chemistry, 1971, 5, 657-668].

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron repulsion tensor
        """
    n = len(basis_functions)
    tensor = qml.math.zeros((n, n, n, n))
    e_calc = qml.math.full((n, n, n, n), np.nan)
    for (i, a), (j, b), (k, c), (l, d) in it.product(enumerate(basis_functions), repeat=4):
        if qml.math.isnan(e_calc[i, j, k, l]):
            args_abcd = []
            if args:
                args_abcd.extend(([arg[i], arg[j], arg[k], arg[l]] for arg in args))
            integral = repulsion_integral(a, b, c, d, normalize=False)(*args_abcd)
            permutations = [(i, j, k, l), (k, l, i, j), (j, i, l, k), (l, k, j, i), (j, i, k, l), (l, k, i, j), (i, j, l, k), (k, l, j, i)]
            o = qml.math.zeros((n, n, n, n))
            for perm in permutations:
                o[perm] = 1.0
                e_calc[perm] = 1.0
            tensor = tensor + integral * o
    return tensor