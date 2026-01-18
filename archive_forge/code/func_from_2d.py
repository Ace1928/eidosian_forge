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
@classmethod
def from_2d(cls, atoms: Atoms, hessian_2d: Union[Sequence[Sequence[Real]], np.ndarray], indices: Sequence[int]=None) -> 'VibrationsData':
    """Instantiate VibrationsData when the Hessian is in a 3Nx3N format

        Args:
            atoms: Equilibrium geometry of vibrating system

            hessian: Second-derivative in energy with respect to
                Cartesian nuclear movements as a (3N, 3N) array.

            indices: Indices of (non-frozen) atoms included in Hessian

        """
    if indices is None:
        indices = range(len(atoms))
    assert indices is not None
    hessian_2d_array = np.asarray(hessian_2d)
    n_atoms = cls._check_dimensions(atoms, hessian_2d_array, indices=indices, two_d=True)
    return cls(atoms, hessian_2d_array.reshape(n_atoms, 3, n_atoms, 3), indices=indices)