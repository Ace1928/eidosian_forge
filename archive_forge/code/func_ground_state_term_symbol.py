from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@property
def ground_state_term_symbol(self):
    """Ground state term symbol
        Selected based on Hund's Rule.
        """
    L_symbols = 'SPDFGHIKLMNOQRTUVWXYZ'
    term_symbols = self.term_symbols
    term_symbol_flat = {term: {'multiplicity': int(term[0]), 'L': L_symbols.index(term[1]), 'J': float(term[2:])} for term in sum(term_symbols, [])}
    multi = [int(item['multiplicity']) for terms, item in term_symbol_flat.items()]
    max_multi_terms = {symbol: item for symbol, item in term_symbol_flat.items() if item['multiplicity'] == max(multi)}
    Ls = [item['L'] for terms, item in max_multi_terms.items()]
    max_L_terms = {symbol: item for symbol, item in term_symbol_flat.items() if item['L'] == max(Ls)}
    J_sorted_terms = sorted(max_L_terms.items(), key=lambda k: k[1]['J'])
    L, v_e = self.valence
    if v_e <= 2 * L + 1:
        return J_sorted_terms[0][0]
    return J_sorted_terms[-1][0]