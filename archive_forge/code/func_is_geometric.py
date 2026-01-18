from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def is_geometric(self, epsilon=1e-06):
    """
        Returns true if all shapes corresponding to this solution have positive
        imaginary part.

        If the solutions are exact, it returns true if one of the corresponding
        numerical solutions is geometric.

        An optional epsilon can be given. An imaginary part of a shape is
        considered positive if it is larger than this epsilon.
        """
    if self._is_numerical:
        for v in self.values():
            if not v.imag() > 0:
                return False
        return True
    else:
        for numerical_sol in self.numerical():
            if numerical_sol.is_geometric(epsilon):
                return True
        return False