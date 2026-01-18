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
def lifrac(e):
    return e.composition.get_atomic_fraction(_working_ion)