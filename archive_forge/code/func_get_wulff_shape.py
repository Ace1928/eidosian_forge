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
def get_wulff_shape(self, material_id):
    """Constructs a Wulff shape for a material.

        Args:
            material_id (str): Materials Project material_id, e.g. 'mp-123'.

        Returns:
            pymatgen.analysis.wulff.WulffShape
        """
    from pymatgen.analysis.wulff import WulffShape
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    structure = self.get_structure_by_material_id(material_id)
    surfaces = self.get_surface_data(material_id)['surfaces']
    lattice = SpacegroupAnalyzer(structure).get_conventional_standard_structure().lattice
    miller_energy_map = {}
    for surf in surfaces:
        miller = tuple(surf['miller_index'])
        if miller not in miller_energy_map or surf['is_reconstructed']:
            miller_energy_map[miller] = surf['surface_energy']
    millers, energies = zip(*miller_energy_map.items())
    return WulffShape(lattice, millers, energies)