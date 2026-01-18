from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def numerical(self):
    """
        Turn exact solutions into numerical solutions using pari. Similar to
        numerical() of PtolemyCoordinates. See help(ptolemy.PtolemyCoordinates)
        for example.
        """
    if self._is_numerical:
        return self
    return ZeroDimensionalComponent([CrossRatios(d, is_numerical=True, manifold_thunk=self._manifold_thunk) for d in _to_numerical(self)])