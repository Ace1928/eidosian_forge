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
@due.dcite(Doi('10.1038/sdata.2016.80'), description='Data Descriptor: Surface energies of elemental crystals. Scientific Data')
def get_surface_data(self, material_id, miller_index=None, inc_structures=False):
    """Gets surface data for a material. Useful for Wulff shapes.

        Reference for surface data:

        Tran, R., Xu, Z., Radhakrishnan, B., Winston, D., Sun, W., Persson, K.
        A., & Ong, S. P. (2016). Data Descriptor: Surface energies of elemental
        crystals. Scientific Data, 3(160080), 1-13.
        https://doi.org/10.1038/sdata.2016.80

        Args:
            material_id (str): Materials Project material_id, e.g. 'mp-123'.
            miller_index (list of integer): The miller index of the surface.
            e.g., [3, 2, 1]. If miller_index is provided, only one dictionary
            of this specific plane will be returned.
            inc_structures (bool): Include final surface slab structures.
                These are unnecessary for Wulff shape construction.

        Returns:
            Surface data for material. Energies are given in SI units (J/m^2).
        """
    req = f'/materials/{material_id}/surfaces'
    if inc_structures:
        req += '?include_structures=true'
    if miller_index:
        surf_data_dict = self._make_request(req)
        surf_list = surf_data_dict['surfaces']
        ucell = self.get_structure_by_material_id(material_id, conventional_unit_cell=True)
        eq_indices = get_symmetrically_equivalent_miller_indices(ucell, miller_index)
        for one_surf in surf_list:
            if tuple(one_surf['miller_index']) in eq_indices:
                return one_surf
        raise ValueError('Bad miller index.')
    return self._make_request(req)