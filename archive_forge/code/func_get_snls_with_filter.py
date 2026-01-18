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
def get_snls_with_filter(self, optimade_filter: str, additional_response_fields: str | list[str] | set[str] | None=None) -> dict[str, dict[str, StructureNL]]:
    """Get structures satisfying a given OPTIMADE filter.

        Args:
            optimade_filter: An OPTIMADE-compliant filter
            additional_response_fields: Any additional fields desired from the OPTIMADE API,

        Returns:
            dict[str, Structure]: keyed by that database provider's id system
        """
    all_snls = {}
    response_fields = self._handle_response_fields(additional_response_fields)
    for identifier, resource in self.resources.items():
        url = urljoin(resource, f'v1/structures?filter={optimade_filter}&response_fields={response_fields!s}')
        try:
            json = self._get_json(url)
            structures = self._get_snls_from_resource(json, url, identifier)
            pbar = tqdm(total=json['meta'].get('data_returned', 0), desc=identifier, initial=len(structures))
            while (next_link := json.get('links', {}).get('next')):
                if isinstance(next_link, dict) and 'href' in next_link:
                    next_link = next_link['href']
                json = self._get_json(next_link)
                additional_structures = self._get_snls_from_resource(json, url, identifier)
                structures.update(additional_structures)
                pbar.update(len(additional_structures))
            if structures:
                all_snls[identifier] = structures
        except Exception as exc:
            _logger.error(f'Could not retrieve required information from provider {identifier} and url={url!r}: {exc}')
    return all_snls