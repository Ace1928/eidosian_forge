from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
def sg_symbol_from_int_number(int_number: int, hexagonal: bool=True) -> str:
    """Obtains a SpaceGroup name from its international number.

    Args:
        int_number (int): International number.
        hexagonal (bool): For rhombohedral groups, whether to return the
            hexagonal setting (default) or rhombohedral setting.

    Returns:
        str: Spacegroup symbol
    """
    syms = []
    for n, v in SYMM_DATA['space_group_encoding'].items():
        if v['int_number'] == int_number:
            syms.append(n)
    if len(syms) == 0:
        raise ValueError('Invalid international number!')
    if len(syms) == 2:
        for sym in syms:
            if 'e' in sym:
                return sym
        if hexagonal:
            syms = list(filter(lambda s: s.endswith('H'), syms))
        else:
            syms = list(filter(lambda s: not s.endswith('H'), syms))
    return syms.pop()