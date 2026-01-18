from __future__ import annotations
import copy
import os
import warnings
from itertools import groupby
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import (
from pymatgen.entries.computed_entries import ComputedStructureEntry, ConstantEnergyAdjustment
from pymatgen.entries.entry_tools import EntrySet
def get_adjustments(self, entry, mixing_state_data: pd.DataFrame | None=None):
    """Returns the corrections applied to a particular entry. Note that get_adjustments is not
        intended to be called directly in the R2SCAN mixing scheme. Call process_entries instead,
        and it will pass the required arguments to get_adjustments.

        Args:
            entry: A ComputedEntry object. The entry must be a member of the list of entries
                used to create mixing_state_data.
            mixing_state_data: A DataFrame containing information about which Entries
                correspond to the same materials, which are stable on the phase diagrams of
                the respective run_types, etc. Can be generated from a list of entries using
                MaterialsProjectDFTMixingScheme.get_mixing_state_data. This argument is included to
                facilitate use of the mixing scheme in high-throughput databases where an alternative
                to get_mixing_state_data is desirable for performance reasons. In general, it should
                always be left at the default value (None) to avoid inconsistencies between the mixing
                state data and the properties of the ComputedStructureEntry.

        Returns:
            [EnergyAdjustment]: Energy adjustments to be applied to entry.

        Raises:
            CompatibilityError if the DFT mixing scheme cannot be applied to the entry.
        """
    adjustments: list[ConstantEnergyAdjustment] = []
    run_type = entry.parameters.get('run_type')
    if mixing_state_data is None:
        raise CompatibilityError('WARNING! `mixing_state_data` DataFrame is None. No energy adjustments will be applied.')
    if not all(mixing_state_data['hull_energy_1'].notna()) and any(mixing_state_data['entry_id_1'].notna()):
        raise CompatibilityError(f'WARNING! {self.run_type_1} entries do not form a complete PhaseDiagram. No energy adjustments will be applied.')
    if run_type not in self.valid_rtypes_1 + self.valid_rtypes_2:
        raise CompatibilityError(f'WARNING! Invalid run_type={run_type!r} for entry {entry.entry_id}. Must be one of {self.valid_rtypes_1 + self.valid_rtypes_2}. This entry will be ignored.')
    if entry.entry_id not in mixing_state_data['entry_id_1'].values and entry.entry_id not in mixing_state_data['entry_id_2'].values:
        raise CompatibilityError(f'WARNING! Discarding {run_type} entry {entry.entry_id} for {entry.formula} because it was not found in the mixing state data. This can occur when there are duplicate structures. In such cases, only the lowest energy entry with that structure appears in the mixing state data.')
    if entry.energy_per_atom not in mixing_state_data['energy_1'].values and entry.energy_per_atom not in mixing_state_data['energy_2'].values:
        raise CompatibilityError(f"WARNING! Discarding {run_type} entry {entry.entry_id} for {entry.formula} because it's energy has been modified since the mixing state data was generated.")
    if all(mixing_state_data[mixing_state_data['is_stable_1']]['entry_id_2'].notna()):
        if run_type in self.valid_rtypes_2:
            return adjustments
        df_slice = mixing_state_data[mixing_state_data['entry_id_1'] == entry.entry_id]
        if df_slice['entry_id_2'].notna().item():
            if df_slice['is_stable_1'].item():
                raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because it is a {self.run_type_1} ground state that matches a {self.run_type_2} material.')
            raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because there is a matching {self.run_type_2} material.')
        hull_energy_1 = df_slice['hull_energy_1'].iloc[0]
        hull_energy_2 = df_slice['hull_energy_2'].iloc[0]
        correction = (hull_energy_2 - hull_energy_1) * entry.composition.num_atoms
        adjustments.append(ConstantEnergyAdjustment(correction, 0.0, name=f'MP {self.run_type_1}/{self.run_type_2} mixing adjustment', cls=self.as_dict(), description=f'Place {self.run_type_1} energy onto the {self.run_type_2} hull'))
        return adjustments
    if any(mixing_state_data[mixing_state_data['is_stable_1']]['entry_id_2'].notna()):
        if run_type in self.valid_rtypes_1:
            df_slice = mixing_state_data[mixing_state_data['entry_id_1'] == entry.entry_id]
            if df_slice['entry_id_2'].notna().item():
                if df_slice['is_stable_1'].item():
                    raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because it is a {self.run_type_1} ground state that matches a {self.run_type_2} material.')
                raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because there is a matching {self.run_type_2} material')
            return adjustments
        df_slice = mixing_state_data[mixing_state_data['formula'] == entry.reduced_formula]
        if any(df_slice[df_slice['is_stable_1']]['entry_id_2'].notna()):
            gs_energy_type_2 = df_slice[df_slice['is_stable_1']]['energy_2'].item()
            e_above_hull = entry.energy_per_atom - gs_energy_type_2
            hull_energy_1 = df_slice['hull_energy_1'].iloc[0]
            correction = (hull_energy_1 + e_above_hull - entry.energy_per_atom) * entry.composition.num_atoms
            adjustments.append(ConstantEnergyAdjustment(correction, 0.0, name=f'MP {self.run_type_1}/{self.run_type_2} mixing adjustment', cls=self.as_dict(), description=f'Place {self.run_type_2} energy onto the {self.run_type_1} hull'))
            return adjustments
        if any(df_slice[df_slice['entry_id_2'] == entry.entry_id]['entry_id_1'].notna()):
            type_1_energy = df_slice[df_slice['entry_id_2'] == entry.entry_id]['energy_1'].iloc[0]
            correction = (type_1_energy - entry.energy_per_atom) * entry.composition.num_atoms
            adjustments.append(ConstantEnergyAdjustment(correction, 0.0, name=f'MP {self.run_type_1}/{self.run_type_2} mixing adjustment', cls=self.as_dict(), description=f'Replace {self.run_type_2} energy with {self.run_type_1} energy'))
            return adjustments
        raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because there is no matching {self.run_type_1} entry and no {self.run_type_2} ground state at this composition.')
    if all(mixing_state_data[mixing_state_data['is_stable_1']]['entry_id_2'].isna()):
        if run_type in self.valid_rtypes_1:
            return adjustments
        raise CompatibilityError(f'Discarding {run_type} entry {entry.entry_id} for {entry.formula} because there are no {self.run_type_2} ground states at this composition.')
    raise CompatibilityError(f'WARNING! If you see this Exception it means you have encounteredan edge case in {type(self).__name__}. Inspect your input carefully and post a bug report.')