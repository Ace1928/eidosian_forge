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
def get_interface_reactions(self, reactant1, reactant2, open_el=None, relative_mu=None, use_hull_energy=False):
    """Gets critical reactions between two reactants.

        Get critical reactions ("kinks" in the mixing ratio where
        reaction products change) between two reactants. See the
        `pymatgen.analysis.interface_reactions` module for more info.

        Args:
            reactant1 (str): Chemical formula for reactant
            reactant2 (str): Chemical formula for reactant
            open_el (str): Element in reservoir available to system
            relative_mu (float): Relative chemical potential of element in
                reservoir with respect to pure substance. Must be non-positive.
            use_hull_energy (bool): Whether to use the convex hull energy for a
            given composition for the reaction energy calculation. If false,
            the energy of the ground state structure will be preferred; if a
            ground state can not be found for a composition, the convex hull
            energy will be used with a warning message.

        Returns:
            list: list of dicts of form {ratio,energy,rxn} where `ratio` is the
                reactant mixing ratio, `energy` is the reaction energy
                in eV/atom, and `rxn` is a
                `pymatgen.analysis.reaction_calculator.Reaction`.
        """
    payload = {'reactants': f'{reactant1} {reactant2}', 'open_el': open_el, 'relative_mu': relative_mu, 'use_hull_energy': use_hull_energy}
    return self._make_request('/interface_reactions', payload=payload, method='POST')