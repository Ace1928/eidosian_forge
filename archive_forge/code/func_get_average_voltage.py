from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
def get_average_voltage(self, min_voltage=None, max_voltage=None):
    """Average voltage for path satisfying between a min and max voltage.

        Args:
            min_voltage (float): The minimum allowable voltage for a given
                step.
            max_voltage (float): The maximum allowable voltage allowable for a
                given step.

        Returns:
            Average voltage in V across the insertion path (a subset of the
            path can be chosen by the optional arguments)
        """
    pairs_in_range = self._select_in_voltage_range(min_voltage, max_voltage)
    if len(pairs_in_range) == 0:
        return 0
    total_cap_in_range = sum((p.mAh for p in pairs_in_range))
    total_edens_in_range = sum((p.mAh * p.voltage for p in pairs_in_range))
    return total_edens_in_range / total_cap_in_range