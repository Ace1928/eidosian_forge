from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def min_voltage(self):
    """Lowest voltage along insertion."""
    return min((p.voltage for p in self.voltage_pairs))