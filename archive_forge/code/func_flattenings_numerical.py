from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def flattenings_numerical(self):
    """
        Turn into numerical solutions and compute flattenings, see
        help(snappy.ptolemy.coordinates.Flattenings)
        Also see numerical()

        Get Ptolemy coordinates.

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Compute a numerical solution

        >>> flattenings = solution.flattenings_numerical()

        Get more information with help(flattenings[0])
        """
    if self._is_numerical:
        branch_factor = 1
        for i in range(1000):
            try:
                d, evenN = _ptolemy_to_cross_ratio(self, branch_factor, self._non_trivial_generalized_obstruction_class, as_flattenings=True)
                return Flattenings(d, manifold_thunk=self._manifold_thunk, evenN=evenN)
            except LogToCloseToBranchCutError:
                branch_factor *= pari('exp(0.0001 * I)')
        raise Exception('Could not find non-ambiguous branch cut for log')
    else:
        return ZeroDimensionalComponent([num.flattenings_numerical() for num in self.numerical()])