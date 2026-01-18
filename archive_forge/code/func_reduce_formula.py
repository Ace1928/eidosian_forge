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
def reduce_formula(sym_amt, iupac_ordering: bool=False) -> tuple[str, float]:
    """Helper method to reduce a sym_amt dict to a reduced formula and factor.

    Args:
        sym_amt (dict): {symbol: amount}.
        iupac_ordering (bool, optional): Whether to order the
            formula by the iupac "electronegativity" series, defined in
            Table VI of "Nomenclature of Inorganic Chemistry (IUPAC
            Recommendations 2005)". This ordering effectively follows
            the groups and rows of the periodic table, except the
            Lanthanides, Actinides and hydrogen. Note that polyanions
            will still be determined based on the true electronegativity of
            the elements.

    Returns:
        tuple[str, float]: reduced formula and factor.
    """
    syms = sorted(sym_amt, key=lambda x: [get_el_sp(x).X, x])
    syms = list(filter(lambda x: abs(sym_amt[x]) > Composition.amount_tolerance, syms))
    factor = 1
    if all((int(i) == i for i in sym_amt.values())):
        factor = abs(gcd(*(int(i) for i in sym_amt.values())))
    poly_anions = []
    if len(syms) >= 3 and get_el_sp(syms[-1]).X - get_el_sp(syms[-2]).X < 1.65:
        poly_sym_amt = {syms[i]: sym_amt[syms[i]] / factor for i in [-2, -1]}
        poly_form, poly_factor = reduce_formula(poly_sym_amt, iupac_ordering=iupac_ordering)
        if poly_factor != 1:
            poly_anions.append(f'({poly_form}){poly_factor}')
    syms = syms[:len(syms) - 2 if poly_anions else len(syms)]
    if iupac_ordering:
        syms = sorted(syms, key=lambda x: [get_el_sp(x).iupac_ordering, x])
    reduced_form: list[str] = []
    for sym in syms:
        norm_amt = sym_amt[sym] * 1.0 / factor
        reduced_form.extend((sym, str(formula_double_format(norm_amt))))
    return (''.join([*reduced_form, *poly_anions]), factor)