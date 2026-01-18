from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def get_reduced_composition_and_factor(self) -> tuple[Composition, float]:
    """Calculates a reduced composition and factor.

        Returns:
            A normalized composition and a multiplicative factor, i.e.,
            Li4Fe4P4O16 returns (Composition("LiFePO4"), 4).
        """
    factor = self.get_reduced_formula_and_factor()[1]
    return (self / factor, factor)