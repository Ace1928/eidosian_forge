from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import BalancedReaction
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
@classmethod
def from_composition_and_entries(cls, comp, entries_in_chemsys, working_ion_symbol='Li', allow_unstable=False) -> Self | None:
    """Convenience constructor to make a ConversionElectrode from a
        composition and all entries in a chemical system.

        Args:
            comp: Starting composition for ConversionElectrode, e.g.,
                Composition("FeF3")
            entries_in_chemsys: Sequence containing all entries in a
               chemical system. E.g., all Li-Fe-F containing entries.
            working_ion_symbol: Element symbol of working ion. Defaults to Li.
            allow_unstable: If True, allow any composition to be used as the
                    starting point of a conversion voltage curve, this is useful
                    for comparing with insertion electrodes
        """
    pd = PhaseDiagram(entries_in_chemsys)
    return cls.from_composition_and_pd(comp, pd, working_ion_symbol, allow_unstable)