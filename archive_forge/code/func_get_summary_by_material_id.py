from __future__ import annotations
import collections
import itertools
import json
import logging
import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING
import requests
from monty.json import MontyDecoder
from pymatgen.core import SETTINGS
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_summary_by_material_id(self, material_id: str, fields: list | None=None) -> dict:
    """
        Get a data corresponding to a material_id.

        Args:
            material_id (str): Materials Project ID (e.g. mp-1234).
            fields (list): Fields to query for. If None (the default), all fields are returned.

        Returns:
            Dict
        """
    get = '_all_fields=True' if fields is None else '_fields=' + ','.join(fields)
    return self.request(f'materials/summary/{material_id}/?{get}')[0]