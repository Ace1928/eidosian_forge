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
def charge_balanced(self) -> bool | None:
    """True if composition is charge balanced, False otherwise. If any oxidation states
        are None, returns None. Use add_charges_from_oxi_state_guesses to assign oxidation
        states to elements.
        """
    warnings.warn('Composition.charge_balanced is experimental and may produce incorrect results. Use with caution and open a GitHub issue pinging @janosh to report bad behavior.')
    if self.charge is None:
        if {getattr(el, 'oxi_state', None) for el in self} == {0}:
            return False
        return None
    return abs(self.charge) < Composition.charge_balanced_tolerance