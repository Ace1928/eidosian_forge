from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import BalancedReaction
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
@property
def initial_comp(self) -> Composition:
    """The pymatgen Composition representation of the initial composition."""
    return Composition(self.initial_comp_formula)