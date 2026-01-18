from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def get_kinks(self) -> list[tuple[int, float, float, Reaction, float]]:
    """
        Finds all the kinks in mixing ratio where reaction products changes
        along the tie-line of composition self.c1 and composition self.c2.

        Returns:
            List object of tuples, each of which contains 5 elements:
            (index, mixing ratio, reaction energy in eV/atom, Reaction object, reaction
            energy per mol of formula in kJ/mol).
        """
    c1_coord = self.pd.pd_coords(self.comp1)
    c2_coord = self.pd.pd_coords(self.comp2)
    n1 = self.comp1.num_atoms
    n2 = self.comp2.num_atoms
    critical_comp = self.pd.get_critical_compositions(self.comp1, self.comp2)
    x_kink, energy_kink, react_kink, energy_per_rxt_formula = ([], [], [], [])
    if (c1_coord == c2_coord).all():
        x_kink = [0, 1]
        energy_kink = [self._get_energy(x) for x in x_kink]
        react_kink = [self._get_reaction(x) for x in x_kink]
        num_atoms = [x * self.comp1.num_atoms + (1 - x) * self.comp2.num_atoms for x in x_kink]
        energy_per_rxt_formula = [energy_kink[idx] * self._get_elem_amt_in_rxn(react_kink[idx]) / num_atoms[idx] * InterfacialReactivity.EV_TO_KJ_PER_MOL for idx in range(2)]
    else:
        for idx in reversed(critical_comp):
            coords = self.pd.pd_coords(idx)
            mixing_ratio = float(np.linalg.norm(coords - c2_coord) / np.linalg.norm(c1_coord - c2_coord))
            mixing_ratio = mixing_ratio * n2 / (n1 + mixing_ratio * (n2 - n1))
            n_atoms = mixing_ratio * self.comp1.num_atoms + (1 - mixing_ratio) * self.comp2.num_atoms
            x_converted = self._convert(mixing_ratio, self.factor1, self.factor2)
            x_kink.append(x_converted)
            normalized_energy = self._get_energy(mixing_ratio)
            energy_kink.append(normalized_energy)
            rxt = self._get_reaction(mixing_ratio)
            react_kink.append(rxt)
            rxt_energy = normalized_energy * self._get_elem_amt_in_rxn(rxt) / n_atoms
            energy_per_rxt_formula.append(rxt_energy * self.EV_TO_KJ_PER_MOL)
    index_kink = range(1, len(critical_comp) + 1)
    return list(zip(index_kink, x_kink, energy_kink, react_kink, energy_per_rxt_formula))