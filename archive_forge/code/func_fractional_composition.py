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
@property
def fractional_composition(self) -> Composition:
    """Returns the normalized composition in which the amounts of each species sum to
        1.
        E.g. "Fe2 O3".fractional_composition = "Fe0.4 O0.6".
        """
    return self / self._n_atoms