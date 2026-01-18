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
def get_ferromagnetic_structure(self, make_primitive: bool=True) -> Structure:
    """Returns a Structure with all magnetic moments positive
        or zero.

        Args:
            make_primitive: Whether to make structure primitive after
                making all magnetic moments positive (Default value = True)

        Returns:
            Structure
        """
    structure = self.structure.copy()
    structure.add_site_property('magmom', [abs(m) for m in self.magmoms])
    if make_primitive:
        structure = structure.get_primitive_structure(use_site_props=True)
    return structure