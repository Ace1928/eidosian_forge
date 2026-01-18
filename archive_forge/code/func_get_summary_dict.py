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
def get_summary_dict(self, print_subelectrodes=True) -> dict:
    """Generate a summary dict.
        Populates the summary dict with the basic information from the parent method then populates more information.
        Since the parent method calls self.get_summary_dict(print_subelectrodes=True) for the subelectrodes.
        The current method will be called from within super().get_summary_dict.

        Args:
            print_subelectrodes: Also print data on all the possible
                subelectrodes.

        Returns:
            A summary of this electrode's properties in dict format.
        """
    dct = super().get_summary_dict(print_subelectrodes=print_subelectrodes)
    chg_comp = self.fully_charged_entry.composition
    dischg_comp = self.fully_discharged_entry.composition
    dct.update({'id_charge': self.fully_charged_entry.entry_id, 'formula_charge': chg_comp.reduced_formula, 'id_discharge': self.fully_discharged_entry.entry_id, 'formula_discharge': dischg_comp.reduced_formula, 'max_instability': self.get_max_instability(), 'min_instability': self.get_min_instability(), 'material_ids': [itr_ent.entry_id for itr_ent in self.get_all_entries()], 'stable_material_ids': [itr_ent.entry_id for itr_ent in self.get_stable_entries()], 'unstable_material_ids': [itr_ent.entry_id for itr_ent in self.get_unstable_entries()]})
    if all(('decomposition_energy' in itr_ent.data for itr_ent in self.get_all_entries())):
        dct.update(stability_charge=self.fully_charged_entry.data['decomposition_energy'], stability_discharge=self.fully_discharged_entry.data['decomposition_energy'], stability_data={itr_ent.entry_id: itr_ent.data['decomposition_energy'] for itr_ent in self.get_all_entries()})
    if all(('muO2' in itr_ent.data for itr_ent in self.get_all_entries())):
        dct.update({'muO2_data': {itr_ent.entry_id: itr_ent.data['muO2'] for itr_ent in self.get_all_entries()}})
    return dct