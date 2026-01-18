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
def get_material_ids(self, formula):
    """
        Get all materials ids for a formula.

        Args:
            formula (str): A formula (e.g., Fe2O3).

        Returns:
            list[str]: all materials ids.
        """
    return [d['material_id'] for d in self.get_summary({'formula': formula}, fields=['material_id'])]