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
def anonymized_formula(self) -> str:
    """An anonymized formula. Unique species are arranged in ordering of
        increasing amounts and assigned ascending alphabets. Useful for
        prototyping formulas. For example, all stoichiometric perovskites have
        anonymized_formula ABC3.
        """
    reduced = self.element_composition
    if all((x == int(x) for x in self.values())):
        reduced /= gcd(*(int(i) for i in self.values()))
    anon = ''
    for elem, amt in zip(string.ascii_uppercase, sorted(reduced.values())):
        if amt == 1:
            amt_str = ''
        elif abs(amt % 1) < 1e-08:
            amt_str = str(int(amt))
        else:
            amt_str = str(amt)
        anon += f'{elem}{amt_str}'
    return anon