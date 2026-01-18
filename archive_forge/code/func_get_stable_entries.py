from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
def get_stable_entries(self, charge_to_discharge=True):
    """Get the stable entries.

        Args:
            charge_to_discharge: order from most charge to most discharged
                state? Default to True.

        Returns:
            A list of stable entries in the electrode, ordered by amount of the
            working ion.
        """
    list_copy = list(self.stable_entries)
    return list_copy if charge_to_discharge else list_copy.reverse()