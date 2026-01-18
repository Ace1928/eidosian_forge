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
def remove_charges(self) -> Composition:
    """Returns a new Composition with charges from each Species removed.

        Returns:
            Composition object without charge decoration, for example
            {"Fe3+": 2.0, "O2-":3.0} becomes {"Fe": 2.0, "O":3.0}
        """
    dct: dict[Element, float] = collections.defaultdict(float)
    for specie, amt in self.items():
        dct[Element(specie.symbol)] += amt
    return Composition(dct)