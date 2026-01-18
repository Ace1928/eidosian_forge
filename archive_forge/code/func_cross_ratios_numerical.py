from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def cross_ratios_numerical(self):
    """
        Turn exact solutions into numerical and then compute cross ratios.
        See numerical() and cross_ratios().
        """
    if self._is_numerical:
        return self.cross_ratios()
    else:
        return ZeroDimensionalComponent([num.cross_ratios() for num in self.numerical()])