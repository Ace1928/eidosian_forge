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
@staticmethod
def indices_from_mask(mask: Union[Sequence[bool], np.ndarray]) -> List[int]:
    """Indices corresponding to boolean mask

        This is provided as a convenience for instantiating VibrationsData with
        a boolean mask. For example, if the Hessian data includes only the H
        atoms in a structure::

          h_mask = atoms.get_chemical_symbols() == 'H'
          vib_data = VibrationsData(atoms, hessian,
                                    VibrationsData.indices_from_mask(h_mask))

        Take care to ensure that the length of the mask corresponds to the full
        number of atoms; this function is only aware of the mask it has been
        given.

        Args:
            mask: a sequence of True, False values

        Returns:
            indices of True elements

        """
    return np.where(mask)[0].tolist()