from __future__ import annotations
import abc
import json
import math
import os
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast
import numpy as np
from monty.json import MontyDecoder, MontyEncoder, MSONable
from scipy.interpolate import interp1d
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.due import Doi, due
@classmethod
def from_pd(cls, pd, temp=300, gibbs_model='SISSO') -> list[Self]:
    """Constructor method for initializing a list of GibbsComputedStructureEntry
        objects from an existing T = 0 K phase diagram composed of
        ComputedStructureEntry objects, as acquired from a thermochemical database;
        (e.g.. The Materials Project).

        Args:
            pd (PhaseDiagram): T = 0 K phase diagram as created in pymatgen. Must
                contain ComputedStructureEntry objects.
            temp (int): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """
    gibbs_entries = []
    for entry in pd.all_entries:
        if entry in pd.el_refs.values() or not entry.composition.is_element:
            gibbs_entries.append(cls(entry.structure, formation_enthalpy_per_atom=pd.get_form_energy_per_atom(entry), temp=temp, correction=0, gibbs_model=gibbs_model, data=entry.data, entry_id=entry.entry_id))
    return gibbs_entries