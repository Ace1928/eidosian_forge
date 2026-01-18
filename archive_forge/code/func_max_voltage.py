from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def max_voltage(self):
    """Highest voltage along insertion."""
    return max((p.voltage for p in self.voltage_pairs))