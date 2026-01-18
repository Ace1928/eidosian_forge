from .cartan_type import Standard_Cartan
from sympy.core.backend import eye
def lie_algebra(self):
    """
        Returns the Lie algebra associated with B_n
        """
    n = self.n
    return 'so(' + str(2 * n) + ')'