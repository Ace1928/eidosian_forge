from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
@property
def calculated_reaction_energy_uncertainty(self) -> float:
    """
        Calculates the uncertainty in the reaction energy based on the uncertainty in the
        energies of the products and reactants.
        """
    calc_energies: dict[Composition, float] = {}
    for entry in self._reactant_entries + self._product_entries:
        comp, factor = entry.composition.get_reduced_composition_and_factor()
        energy_ufloat = ufloat(entry.energy, entry.correction_uncertainty)
        calc_energies[comp] = min(calc_energies.get(comp, float('inf')), energy_ufloat / factor)
    return self.calculate_energy(calc_energies).std_dev