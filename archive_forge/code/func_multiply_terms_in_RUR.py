from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def multiply_terms_in_RUR(self):
    """
        If a cross ratio is given as Rational Univariate Representation
        with numerator and denominator being a product, multiply the terms and
        return the result.

        See multiply_terms of RUR.

        This loses information about how the numerator and denominator are
        factorised.
        """
    return CrossRatios(_apply_to_RURs(self, RUR.multiply_terms), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)