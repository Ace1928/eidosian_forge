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
@staticmethod
def print_periodic_table(filter_function: Callable | None=None) -> None:
    """A pretty ASCII printer for the periodic table, based on some
        filter_function.

        Args:
            filter_function: A filtering function taking an Element as input
                and returning a boolean. For example, setting
                filter_function = lambda el: el.X > 2 will print a periodic
                table containing only elements with Pauling electronegativity > 2.
        """
    for row in range(1, 10):
        row_str = []
        for group in range(1, 19):
            try:
                el = Element.from_row_and_group(row, group)
            except ValueError:
                el = None
            if el and (not filter_function or filter_function(el)):
                row_str.append(f'{el.symbol:3s}')
            else:
                row_str.append('   ')
        print(' '.join(row_str))