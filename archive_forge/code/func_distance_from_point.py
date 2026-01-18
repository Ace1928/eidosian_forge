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
def distance_from_point(self, pt) -> float:
    """Returns distance between the site and a point in space.

        Args:
            pt: Cartesian coordinates of point.

        Returns:
            float: distance
        """
    return float(np.linalg.norm(np.array(pt) - self.coords))