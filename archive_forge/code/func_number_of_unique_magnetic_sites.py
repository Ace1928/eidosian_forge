from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
def number_of_unique_magnetic_sites(self, symprec: float=0.001, angle_tolerance: float=5) -> int:
    """
        Args:
            symprec: same as in SpacegroupAnalyzer (Default value = 1e-3)
            angle_tolerance: same as in SpacegroupAnalyzer (Default value = 5).

        Returns:
            int: Number of symmetrically-distinct magnetic sites present in structure.
        """
    structure = self.get_nonmagnetic_structure()
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    symm_structure = sga.get_symmetrized_structure()
    num_unique_mag_sites = 0
    for group_of_sites in symm_structure.equivalent_sites:
        if group_of_sites[0].specie in self.types_of_magnetic_species:
            num_unique_mag_sites += 1
    return num_unique_mag_sites