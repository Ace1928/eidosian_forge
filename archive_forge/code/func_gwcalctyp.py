from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
@property
def gwcalctyp(self):
    """Returns the value of the gwcalctyp input variable."""
    dig0 = str(self._SIGMA_TYPES[self.type])
    dig1 = str(self._SC_MODES[self.sc_mode])
    return dig1.strip() + dig0.strip()