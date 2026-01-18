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
def contains_element_type(self, category: str) -> bool:
    """Check if Composition contains any elements matching a given category.

        Args:
            category (str): one of "noble_gas", "transition_metal",
                "post_transition_metal", "rare_earth_metal", "metal", "metalloid",
                "alkali", "alkaline", "halogen", "chalcogen", "lanthanoid",
                "actinoid", "quadrupolar", "s-block", "p-block", "d-block", "f-block"

        Returns:
            bool: True if any elements in Composition match category, otherwise False
        """
    allowed_categories = [category.value for category in ElementType]
    if category not in allowed_categories:
        raise ValueError(f'Invalid category={category!r}, pick from {allowed_categories}')
    if 'block' in category:
        return any((category[0] in el.block for el in self.elements))
    return any((getattr(el, f'is_{category}') for el in self.elements))