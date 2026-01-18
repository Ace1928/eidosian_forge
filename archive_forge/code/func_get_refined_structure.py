from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def get_refined_structure(self, keep_site_properties=False):
    """Get the refined structure based on detected symmetry. The refined structure is
        a *conventional* cell setting with atoms moved to the expected symmetry positions.

        Args:
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            Refined structure.
        """
    lattice, scaled_positions, numbers = spglib.refine_cell(self._cell, self._symprec, self._angle_tol)
    species = [self._unique_species[i - 1] for i in numbers]
    if keep_site_properties:
        site_properties = {}
        for k, v in self._site_props.items():
            site_properties[k] = [v[i - 1] for i in numbers]
    else:
        site_properties = None
    struct = Structure(lattice, species, scaled_positions, site_properties=site_properties)
    return struct.get_sorted_structure()