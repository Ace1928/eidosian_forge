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
def iter_animated_mode(self, mode_index: int, temperature: float=units.kB * 300, frames: int=30) -> Iterator[Atoms]:
    """Obtain animated mode as a series of Atoms

        Args:
            mode_index: Selection of mode to animate
            temperature: In energy units - use units.kB * T_IN_KELVIN
            frames: number of image frames in animation

        Yields:
            Displaced atoms following vibrational mode

        """
    mode = self.get_modes(all_atoms=True)[mode_index] * sqrt(temperature / abs(self.get_energies()[mode_index]))
    for phase in np.linspace(0, 2 * pi, frames, endpoint=False):
        atoms = self.get_atoms()
        atoms.positions += sin(phase) * mode
        yield atoms