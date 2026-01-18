from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.matrixutils import (
def cache_matrix(self, name, m):
    """Cache a matrix by its name.

        Parameters
        ----------
        name : str
            A descriptive name for the matrix, like "identity2".
        m : list of lists
            The raw matrix data as a SymPy Matrix.
        """
    try:
        self._sympy_matrix(name, m)
    except ImportError:
        pass
    try:
        self._numpy_matrix(name, m)
    except ImportError:
        pass
    try:
        self._scipy_sparse_matrix(name, m)
    except ImportError:
        pass