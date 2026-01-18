from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
def basic_root(self, i, j):
    """
        This is a method just to generate roots
        with a -1 in the ith position and a 1
        in the jth position.

        """
    root = [0] * 8
    root[i] = -1
    root[j] = 1
    return root