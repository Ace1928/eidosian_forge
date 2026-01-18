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
def get_snls(self, elements: list[str] | str | None=None, nelements: int | None=None, nsites: int | None=None, chemical_formula_anonymous: str | None=None, chemical_formula_hill: str | None=None, additional_response_fields: str | list[str] | set[str] | None=None) -> dict[str, dict[str, StructureNL]]:
    """Retrieve StructureNL from OPTIMADE providers.

        A StructureNL is an object provided by pymatgen which combines Structure with
        associated metadata, such as the URL is was downloaded from and any additional namespaced
        data.

        Not all functionality of OPTIMADE is currently exposed in this convenience method. To
        use a custom filter, call get_structures_with_filter().

        Args:
            elements: List of elements
            nelements: Number of elements, e.g. 4 or [2, 5] for the range >=2 and <=5
            nsites: Number of sites, e.g. 4 or [2, 5] for the range >=2 and <=5
            chemical_formula_anonymous: The desired chemical formula in OPTIMADE anonymous formula format
            (NB. The ordering is reversed from the pymatgen format, e.g., pymatgen "ABC2" should become "A2BC").
            chemical_formula_hill: The desired chemical formula in the OPTIMADE take on the Hill formula format.
            (NB. Again, this is different from the pymatgen format, as the OPTIMADE version is a reduced chemical
            formula simply using the IUPAC/Hill ordering.)
            additional_response_fields: Any additional fields desired from the OPTIMADE API,
            these will be stored under the `'_optimade'` key in each `StructureNL.data` dictionary.

        Returns:
            dict[str, StructureNL]: keyed by that database provider's id system
        """
    optimade_filter = self._build_filter(elements=elements, nelements=nelements, nsites=nsites, chemical_formula_anonymous=chemical_formula_anonymous, chemical_formula_hill=chemical_formula_hill)
    return self.get_snls_with_filter(optimade_filter, additional_response_fields=additional_response_fields)