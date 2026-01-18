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
def query_snl(self, criteria):
    """Query for submitted SNLs.

        Note:
            As of now, this MP REST feature is open only to a select group of
            users. Opening up submissions to all users is being planned for the future.

        Args:
            criteria (dict): Query criteria.

        Returns:
            A dict, with a list of submitted SNLs in the "response" key.

        Raises:
            MPRestError
        """
    payload = {'criteria': json.dumps(criteria)}
    response = self.session.post(f'{self.preamble}/snl/query', data=payload)
    if response.status_code in [200, 400]:
        response = json.loads(response.text)
        if response['valid_response']:
            if response.get('warning'):
                warnings.warn(response['warning'])
            return response['response']
        raise MPRestError(response['error'])
    raise MPRestError(f'REST error with status code {response.status_code} and error {response.text}')