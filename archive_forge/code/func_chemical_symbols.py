from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
@lazy_property
def chemical_symbols(self):
    """Chemical symbols char [number of atom species][symbol length]."""
    charr = self.read_value('chemical_symbols')
    symbols = []
    for v in charr:
        s = ''.join((c.decode('utf-8') for c in v))
        symbols.append(s.strip())
    return symbols