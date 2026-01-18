from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def get_cohesive_energy(self, material_id, per_atom=False):
    """Gets the cohesive for a material (eV per formula unit). Cohesive energy
            is defined as the difference between the bulk energy and the sum of
            total DFT energy of isolated atoms for atom elements in the bulk.

        Args:
            material_id (str): Materials Project material_id, e.g. 'mp-123'.
            per_atom (bool): Whether or not to return cohesive energy per atom

        Returns:
            Cohesive energy (eV).
        """
    entry = self.get_entry_by_material_id(material_id)
    ebulk = entry.energy / entry.composition.get_integer_formula_and_factor()[1]
    comp_dict = entry.composition.reduced_composition.as_dict()
    isolated_atom_e_sum, n = (0, 0)
    for el in comp_dict:
        e = self._make_request(f'/element/{el}/tasks/isolated_atom', mp_decode=False)[0]
        isolated_atom_e_sum += e['output']['final_energy_per_atom'] * comp_dict[el]
        n += comp_dict[el]
    ecoh_per_formula = isolated_atom_e_sum - ebulk
    return ecoh_per_formula / n if per_atom else ecoh_per_formula