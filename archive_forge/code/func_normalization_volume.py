from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@property
def normalization_volume(self):
    """Returns: Mass used for normalization. This is the vol of the discharged
        electrode of the last voltage pair.
        """
    return self.voltage_pairs[-1].vol_discharge