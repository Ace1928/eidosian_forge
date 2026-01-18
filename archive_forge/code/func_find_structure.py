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
def find_structure(self, filename_or_structure):
    """Finds matching structures on the Materials Project site.

        Args:
            filename_or_structure: filename or Structure object

        Returns:
            A list of matching materials project ids for structure.

        Raises:
            MPRestError
        """
    if isinstance(filename_or_structure, str):
        struct = Structure.from_file(filename_or_structure)
    elif isinstance(filename_or_structure, Structure):
        struct = filename_or_structure
    else:
        raise MPRestError('Provide filename or Structure object.')
    payload = {'structure': json.dumps(struct.as_dict(), cls=MontyEncoder)}
    response = self.session.post(f'{self.preamble}/find_structure', data=payload)
    if response.status_code in [200, 400]:
        response = json.loads(response.text, cls=MontyDecoder)
        if response['valid_response']:
            return response['response']
        raise MPRestError(response['error'])
    raise MPRestError(f'REST error with status code {response.status_code} and error {response.text}')