from __future__ import annotations
import logging
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm
from pymatgen.core import DummySpecies, Structure
from pymatgen.util.due import Doi, due
from pymatgen.util.provenance import StructureNL
def get_structures_with_filter(self, optimade_filter: str) -> dict[str, dict[str, Structure]]:
    """Get structures satisfying a given OPTIMADE filter.

        Args:
            optimade_filter: An OPTIMADE-compliant filter

        Returns:
            dict[str, Structure]: keyed by that database provider's id system
        """
    all_snls = self.get_snls_with_filter(optimade_filter)
    all_structures = {}
    for identifier, snls_dict in all_snls.items():
        all_structures[identifier] = {k: snl.structure for k, snl in snls_dict.items()}
    return all_structures