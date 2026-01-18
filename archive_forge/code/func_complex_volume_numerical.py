from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def complex_volume_numerical(self, drop_negative_vols=False, with_modulo=False):
    """
        Turn into (Galois conjugate) numerical solutions and compute complex
        volumes. If already numerical, return the volume.

        Complex volume is defined up to i*pi**2/6.

        See numerical(). If drop_negative_vols = True is given as optional
        argument, only return complex volumes with non-negative real part.
        """
    if self._is_numerical:
        return self.flattenings_numerical().complex_volume(with_modulo=with_modulo)
    else:
        cvols = ZeroDimensionalComponent([num.flattenings_numerical().complex_volume(with_modulo=with_modulo) for num in self.numerical()])
        if drop_negative_vols:
            return [cvol for cvol in cvols if cvol.real() > -1e-12]
        return cvols