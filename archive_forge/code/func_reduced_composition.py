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
def reduced_composition(self) -> Composition:
    """Returns the reduced composition, i.e. amounts normalized by greatest common denominator.
        E.g. "Fe4 P4 O16".reduced_composition = "Fe P O4".
        """
    return self.get_reduced_composition_and_factor()[0]