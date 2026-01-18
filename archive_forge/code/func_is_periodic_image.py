from __future__ import annotations
import collections
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.util.coord import pbc_diff
def is_periodic_image(self, other: PeriodicSite, tolerance: float=1e-08, check_lattice: bool=True) -> bool:
    """Returns True if sites are periodic images of each other.

        Args:
            other (PeriodicSite): Other site
            tolerance (float): Tolerance to compare fractional coordinates
            check_lattice (bool): Whether to check if the two sites have the
                same lattice.

        Returns:
            bool: True if sites are periodic images of each other.
        """
    if check_lattice and self.lattice != other.lattice:
        return False
    if self.species != other.species:
        return False
    frac_diff = pbc_diff(self.frac_coords, other.frac_coords, self.lattice.pbc)
    return np.allclose(frac_diff, [0, 0, 0], atol=tolerance)