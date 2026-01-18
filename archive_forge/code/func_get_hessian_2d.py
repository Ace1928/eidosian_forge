import collections
from math import sin, pi, sqrt
from numbers import Real, Integral
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.utils import jsonable, lazymethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection
def get_hessian_2d(self) -> np.ndarray:
    """Get the Hessian as a 2-D array

        This format may be preferred for use with standard linear algebra
        functions

        Returns:
            array with shape (n_atoms * 3, n_atoms * 3) where the elements are
            ordered by atom and Cartesian direction

            [[at1x_at1x, at1x_at1y, at1x_at1z, at1x_at2x, ...],
             [at1y_at1x, at1y_at1y, at1y_at1z, at1y_at2x, ...],
             [at1z_at1x, at1z_at1y, at1z_at1z, at1z_at2x, ...],
             [at2x_at1x, at2x_at1y, at2x_at1z, at2x_at2x, ...],
             ...]

            e.g. the element h[2, 3] gives a harmonic force exerted on
            atoms[1] in the x-direction in response to a movement in the
            z-direction of atoms[0]

        """
    return self._hessian2d.copy()