from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_oxi_state_decorated_structure(self, structure: Structure):
    """
        Get an oxidation state decorated structure. This currently works only
        for ordered structures only.

        Args:
            structure: Structure to analyze

        Returns:
            A modified structure that is oxidation state decorated.

        Raises:
            ValueError if the valences cannot be determined.
        """
    struct = structure.copy()
    if struct.is_ordered:
        valences = self.get_valences(struct)
        struct.add_oxidation_state_by_site(valences)
    else:
        valences = self.get_valences(struct)
        struct = add_oxidation_state_by_site_fraction(struct, valences)
    return struct