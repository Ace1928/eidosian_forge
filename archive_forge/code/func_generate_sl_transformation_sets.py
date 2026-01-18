from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
def generate_sl_transformation_sets(self, film_area, substrate_area):
    """
        Generates transformation sets for film/substrate pair given the
        area of the unit cell area for the film and substrate. The
        transformation sets map the film and substrate unit cells to super
        lattices with a maximum area

        Args:
            film_area (int): the unit cell area for the film
            substrate_area (int): the unit cell area for the substrate

        Returns:
            transformation_sets: a set of transformation_sets defined as:
                1.) the transformation matrices for the film to create a
                super lattice of area i*film area
                2.) the transformation matrices for the substrate to create
                a super lattice of area j*film area.
        """
    transformation_indices = [(ii, jj) for ii in range(1, int(np.ceil(self.max_area / film_area))) for jj in range(1, int(np.ceil(self.max_area / substrate_area))) if np.absolute(film_area / substrate_area - float(jj) / ii) < self.max_area_ratio_tol] + [(ii, jj) for ii in range(1, int(np.ceil(self.max_area / film_area))) for jj in range(1, int(np.ceil(self.max_area / substrate_area))) if np.absolute(substrate_area / film_area - float(ii) / jj) < self.max_area_ratio_tol]
    transformation_indices = list(set(transformation_indices))
    for ii, jj in sorted(transformation_indices, key=lambda x: x[0] * x[1]):
        yield (gen_sl_transform_matrices(ii), gen_sl_transform_matrices(jj))