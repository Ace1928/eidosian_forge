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
def calculated_reaction_energy(self) -> float:
    """
        Returns:
            float: The calculated reaction energy.
        """
    calc_energies: dict[Composition, float] = {}
    for entry in self._reactant_entries + self._product_entries:
        comp, factor = entry.composition.get_reduced_composition_and_factor()
        calc_energies[comp] = min(calc_energies.get(comp, float('inf')), entry.energy / factor)
    return self.calculate_energy(calc_energies)