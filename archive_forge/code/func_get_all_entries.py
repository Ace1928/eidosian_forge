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
def get_all_entries(self, charge_to_discharge=True):
    """Return all entries input for the electrode.

        Args:
            charge_to_discharge:
                order from most charge to most discharged state? Defaults to
                True.

        Returns:
            A list of all entries in the electrode (both stable and unstable),
            ordered by amount of the working ion.
        """
    all_entries = list(self.get_stable_entries())
    all_entries.extend(self.get_unstable_entries())
    all_entries = sorted(all_entries, key=lambda e: e.composition.get_atomic_fraction(self.working_ion))
    return all_entries if charge_to_discharge else all_entries.reverse()