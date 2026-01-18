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
def distance_and_image(self, other: PeriodicSite, jimage: ArrayLike | None=None) -> tuple[float, np.ndarray]:
    """Gets distance and instance between two sites assuming periodic boundary
        conditions. If the index jimage of two sites atom j is not specified it
        selects the j image nearest to the i atom and returns the distance and
        jimage indices in terms of lattice vector translations. If the index
        jimage of atom j is specified it returns the distance between the ith
        atom and the specified jimage atom, the given jimage is also returned.

        Args:
            other (PeriodicSite): Other site to get distance from.
            jimage (3x1 array): Specific periodic image in terms of lattice
                translations, e.g., [1,0,0] implies to take periodic image
                that is one a-lattice vector away. If jimage is None,
                the image that is nearest to the site is found.

        Returns:
            tuple[float, np.ndarray]: distance and periodic lattice translations (jimage)
                of the other site for which the distance applies.
        """
    return self.distance_and_image_from_frac_coords(other.frac_coords, jimage)