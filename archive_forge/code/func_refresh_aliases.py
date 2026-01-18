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
def refresh_aliases(self, providers_url='https://providers.optimade.org/providers.json'):
    """Updates available OPTIMADE structure resources based on the current list of OPTIMADE
        providers.
        """
    json = self._get_json(providers_url)
    providers_from_url = {entry['id']: entry['attributes']['base_url'] for entry in json['data'] if entry['attributes']['base_url']}
    structure_providers = {}
    for provider, provider_link in providers_from_url.items():
        structure_providers.update(self._parse_provider(provider, provider_link))
    self.aliases = {alias: provider.base_url for alias, provider in structure_providers.items()}