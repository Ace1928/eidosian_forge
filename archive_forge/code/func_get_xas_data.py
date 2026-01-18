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
def get_xas_data(self, material_id, absorbing_element):
    """Get X-ray absorption spectroscopy data for absorbing element in the
        structure corresponding to a material_id. Only X-ray Absorption Near Edge
        Structure (XANES) for K-edge is supported.

        REST Endpoint:
        https://materialsproject.org/materials/<mp-id>/xas/<absorbing_element>.

        Args:
            material_id (str): E.g., mp-1143 for Al2O3
            absorbing_element (str): The absorbing element in the corresponding
                structure. E.g., Al in Al2O3
        """
    element_list = self.get_data(material_id, prop='elements')[0]['elements']
    if absorbing_element not in element_list:
        raise ValueError(f'{absorbing_element} element not contained in corresponding structure with mp_id: {material_id}')
    data = self._make_request(f'/materials/{material_id}/xas/{absorbing_element}', mp_decode=False)
    return data[0]