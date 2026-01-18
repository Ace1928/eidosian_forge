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
def from_composition_and_pd(cls, comp, pd: PhaseDiagram, working_ion_symbol: str='Li', allow_unstable: bool=False) -> Self | None:
    """Convenience constructor to make a ConversionElectrode from a
        composition and a phase diagram.

        Args:
            comp: Starting composition for ConversionElectrode, e.g.,
                Composition("FeF3")
            pd: A PhaseDiagram of the relevant system (e.g., Li-Fe-F)
            working_ion_symbol: Element symbol of working ion. Defaults to Li.
            allow_unstable: Allow compositions that are unstable
        """
    working_ion = Element(working_ion_symbol)
    entry = working_ion_entry = None
    for ent in pd.stable_entries:
        if ent.reduced_formula == comp.reduced_formula:
            entry = ent
        elif ent.is_element and ent.reduced_formula == working_ion_symbol:
            working_ion_entry = ent
    if not allow_unstable and (not entry):
        raise ValueError(f'Not stable compound found at composition {comp}.')
    profile = pd.get_element_profile(working_ion, comp)
    profile.reverse()
    if len(profile) < 2:
        return None
    assert working_ion_entry is not None
    working_ion_symbol = working_ion_entry.elements[0].symbol
    normalization_els = {el: amt for el, amt in comp.items() if el != Element(working_ion_symbol)}
    framework = comp.as_dict()
    if working_ion_symbol in framework:
        framework.pop(working_ion_symbol)
    framework = Composition(framework)
    v_pairs: list[ConversionVoltagePair] = [ConversionVoltagePair.from_steps(profile[i], profile[i + 1], normalization_els, framework_formula=framework.reduced_formula) for i in range(len(profile) - 1)]
    return cls(voltage_pairs=v_pairs, working_ion_entry=working_ion_entry, initial_comp_formula=comp.reduced_formula, framework_formula=framework.reduced_formula)