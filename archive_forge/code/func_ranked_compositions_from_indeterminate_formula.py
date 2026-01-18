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
@staticmethod
def ranked_compositions_from_indeterminate_formula(fuzzy_formula: str, lock_if_strict: bool=True) -> list[Composition]:
    """Takes in a formula where capitalization might not be correctly entered,
        and suggests a ranked list of potential Composition matches.
        Author: Anubhav Jain.

        Args:
            fuzzy_formula (str): A formula string, such as "co2o3" or "MN",
                that may or may not have multiple interpretations
            lock_if_strict (bool): If true, a properly entered formula will
                only return the one correct interpretation. For example,
                "Co1" will only return "Co1" if true, but will return both
                "Co1" and "C1 O1" if false.

        Returns:
            A ranked list of potential Composition matches
        """
    if lock_if_strict:
        try:
            comp = Composition(fuzzy_formula)
            return [comp]
        except ValueError:
            pass
    all_matches = Composition._comps_from_fuzzy_formula(fuzzy_formula)
    uniq_matches = list(set(all_matches))
    ranked_matches = sorted(uniq_matches, key=lambda match: (match[1], match[0]), reverse=True)
    return [m[0] for m in ranked_matches]