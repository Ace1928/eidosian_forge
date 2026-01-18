from __future__ import annotations
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from monty.serialization import loadfn
from ruamel import yaml
from scipy.optimize import curve_fit
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.structure_analyzer import sulfide_type
from pymatgen.core import Composition, Element
def compute_corrections(self, exp_entries: list, calc_entries: dict) -> dict:
    """Computes the corrections and fills in correction, corrections_std_error, and corrections_dict.

        Args:
            exp_entries: list of dictionary objects with the following keys/values:
                    {"formula": chemical formula, "exp energy": formation energy in eV/formula unit,
                    "uncertainty": uncertainty in formation energy}
            calc_entries: dictionary of computed entries, of the form {chemical formula: ComputedEntry}

        Raises:
            ValueError: calc_compounds is missing an entry
        """
    self.exp_compounds = exp_entries
    self.calc_compounds = calc_entries
    self.names: list[str] = []
    self.diffs: list[float] = []
    self.coeff_mat: list[list[float]] = []
    self.exp_uncer: list[float] = []
    for entry in self.calc_compounds.values():
        entry.correction = 0
    for cmpd_info in self.exp_compounds:
        name = Composition(cmpd_info['formula']).reduced_formula
        allow = True
        compound = self.calc_compounds.get(name)
        if not compound:
            warnings.warn(f'Compound {name} is not found in provided computed entries and is excluded from the fit')
            continue
        relative_uncertainty = abs(cmpd_info['uncertainty'] / cmpd_info['exp energy'])
        if relative_uncertainty > self.max_error:
            allow = False
            warnings.warn(f'Compound {name} is excluded from the fit due to high experimental uncertainty ({relative_uncertainty:.1%})')
        for anion in self.exclude_polyanions:
            if anion in name or anion in cmpd_info['formula']:
                allow = False
                warnings.warn(f'Compound {name} contains the polyanion={anion!r} and is excluded from the fit')
                break
        if isinstance(self.allow_unstable, float):
            try:
                eah = compound.data['e_above_hull']
            except KeyError:
                raise ValueError('Missing e above hull data')
            if eah > self.allow_unstable:
                allow = False
                warnings.warn(f'Compound {name} is unstable and excluded from the fit (e_above_hull = {eah})')
        if allow:
            comp = Composition(name)
            elems = list(comp.as_dict())
            reactants = []
            for elem in elems:
                try:
                    elem_name = Composition(elem).reduced_formula
                    reactants.append(self.calc_compounds[elem_name])
                except KeyError:
                    raise ValueError('Computed entries missing ' + elem)
            rxn = ComputedReaction(reactants, [compound])
            rxn.normalize_to(comp)
            energy = rxn.calculated_reaction_energy
            coeff = []
            for specie in self.species:
                if specie == 'oxide':
                    if compound.data['oxide_type'] == 'oxide':
                        coeff.append(comp['O'])
                        self.oxides.append(name)
                    else:
                        coeff.append(0)
                elif specie == 'peroxide':
                    if compound.data['oxide_type'] == 'peroxide':
                        coeff.append(comp['O'])
                        self.peroxides.append(name)
                    else:
                        coeff.append(0)
                elif specie == 'superoxide':
                    if compound.data['oxide_type'] == 'superoxide':
                        coeff.append(comp['O'])
                        self.superoxides.append(name)
                    else:
                        coeff.append(0)
                elif specie == 'S':
                    if Element('S') in comp:
                        sf_type = 'sulfide'
                        if compound.data.get('sulfide_type'):
                            sf_type = compound.data['sulfide_type']
                        elif hasattr(compound, 'structure'):
                            sf_type = sulfide_type(compound.structure)
                        if sf_type == 'sulfide':
                            coeff.append(comp['S'])
                            self.sulfides.append(name)
                        else:
                            coeff.append(0)
                    else:
                        coeff.append(0)
                else:
                    try:
                        coeff.append(comp[specie])
                    except ValueError:
                        raise ValueError(f"We can't detect this specie={specie!r} in name={name!r}")
            self.names.append(name)
            self.diffs.append((cmpd_info['exp energy'] - energy) / comp.num_atoms)
            self.coeff_mat.append([i / comp.num_atoms for i in coeff])
            self.exp_uncer.append(cmpd_info['uncertainty'] / comp.num_atoms)
    sigma = np.array(self.exp_uncer)
    sigma[sigma == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_uncert = np.nanmean(sigma)
    sigma = np.where(np.isnan(sigma), mean_uncert, sigma)
    if np.isnan(mean_uncert):
        p_opt, self.pcov = curve_fit(lambda x, *m: np.dot(x, m), self.coeff_mat, self.diffs, p0=np.ones(len(self.species)))
    else:
        p_opt, self.pcov = curve_fit(lambda x, *m: np.dot(x, m), self.coeff_mat, self.diffs, p0=np.ones(len(self.species)), sigma=sigma, absolute_sigma=True)
    self.corrections = p_opt.tolist()
    self.corrections_std_error = np.sqrt(np.diag(self.pcov)).tolist()
    for idx, specie in enumerate(self.species):
        self.corrections_dict[specie] = (round(self.corrections[idx], 3), round(self.corrections_std_error[idx], 4))
    self.corrections_dict['ozonide'] = (0, 0)
    return self.corrections_dict