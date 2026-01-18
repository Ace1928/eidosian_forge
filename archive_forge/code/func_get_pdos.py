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
def get_pdos(self) -> DOSCollection:
    """Phonon DOS, including atomic contributions"""
    energies = self.get_energies()
    masses = self._atoms[self.get_mask()].get_masses()
    vectors = self.get_modes() / masses[np.newaxis, :, np.newaxis] ** (-0.5)
    all_weights = (np.linalg.norm(vectors, axis=-1) ** 2).T
    mask = self.get_mask()
    all_info = [{'index': i, 'symbol': a.symbol} for i, a in enumerate(self._atoms) if mask[i]]
    return DOSCollection([RawDOSData(energies, weights, info=info) for weights, info in zip(all_weights, all_info)])