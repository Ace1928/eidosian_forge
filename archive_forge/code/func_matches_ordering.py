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
def matches_ordering(self, other: Structure) -> bool:
    """Compares the magnetic orderings of one structure with another.

        Args:
            other: Structure to compare

        Returns:
            bool: True if magnetic orderings match, False otherwise
        """
    cmag_analyzer = CollinearMagneticStructureAnalyzer(self.structure, overwrite_magmom_mode='normalize').get_structure_with_spin()
    b_positive = CollinearMagneticStructureAnalyzer(other, overwrite_magmom_mode='normalize', make_primitive=False)
    b_negative = b_positive.structure.copy()
    b_negative.add_site_property('magmom', -np.array(b_negative.site_properties['magmom']))
    analyzer = CollinearMagneticStructureAnalyzer(b_negative, overwrite_magmom_mode='normalize', make_primitive=False)
    b_positive = b_positive.get_structure_with_spin()
    analyzer = analyzer.get_structure_with_spin()
    return cmag_analyzer.matches(b_positive) or cmag_analyzer.matches(analyzer)