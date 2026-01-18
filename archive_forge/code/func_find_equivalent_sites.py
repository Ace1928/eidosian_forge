from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from tabulate import tabulate
from pymatgen.core.structure import PeriodicSite, Structure
def find_equivalent_sites(self, site: PeriodicSite) -> list[PeriodicSite]:
    """Finds all symmetrically equivalent sites for a particular site.

        Args:
            site (PeriodicSite): A site in the structure

        Raises:
            ValueError: if site is not in the structure.

        Returns:
            list[PeriodicSite]: all symmetrically equivalent sites.
        """
    for sites in self.equivalent_sites:
        if site in sites:
            return sites
    raise ValueError('Site not in structure')