import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def attraction(*args):
    """Construct the electron-nuclear attraction matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron-nuclear attraction matrix
        """
    n = len(basis_functions)
    matrix = qml.math.zeros((n, n))
    for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
        integral = 0
        if args:
            args_ab = []
            if r.requires_grad:
                args_ab.extend(([arg[i], arg[j]] for arg in args[1:]))
            else:
                args_ab.extend(([arg[i], arg[j]] for arg in args))
            for k, c in enumerate(r):
                if c.requires_grad:
                    args_ab = [args[0][k]] + args_ab
                integral = integral - charges[k] * attraction_integral(c, a, b, normalize=False)(*args_ab)
                if c.requires_grad:
                    args_ab = args_ab[1:]
        else:
            for k, c in enumerate(r):
                integral = integral - charges[k] * attraction_integral(c, a, b, normalize=False)()
        o = qml.math.zeros((n, n))
        o[i, j] = o[j, i] = 1.0
        matrix = matrix + integral * o
    return matrix