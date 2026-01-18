from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def equivalent_site_index_and_transform(self, psite):
    """Get the equivalent site and corresponding symmetry+translation transformations.

        Args:
            psite: Periodic site.

        Returns:
            Equivalent site in the unit cell, translations and symmetry transformation.
        """
    try:
        isite = self.structure_environments.structure.index(psite)
    except ValueError:
        try:
            uc_psite = psite.to_unit_cell()
            isite = self.structure_environments.structure.index(uc_psite)
        except ValueError:
            for isite2, site2 in enumerate(self.structure_environments.structure):
                if psite.is_periodic_image(site2):
                    isite = isite2
                    break
    this_site = self.structure_environments.structure[isite]
    dthis_site = psite.frac_coords - this_site.frac_coords
    equiv_site = self.structure_environments.structure[self.structure_environments.sites_map[isite]].to_unit_cell()
    dequivsite = self.structure_environments.structure[self.structure_environments.sites_map[isite]].frac_coords - equiv_site.frac_coords
    found = False
    tolerances = [1e-08, 1e-07, 1e-06, 1e-05, 0.0001]
    for tolerance in tolerances:
        for sym_op in self.symops:
            new_site = PeriodicSite(equiv_site._species, sym_op.operate(equiv_site.frac_coords), equiv_site._lattice)
            if new_site.is_periodic_image(this_site, tolerance=tolerance):
                sym_trafo = sym_op
                d_this_site2 = this_site.frac_coords - new_site.frac_coords
                found = True
                break
        if not found:
            sym_ops = [SymmOp.from_rotation_and_translation()]
            for sym_op in sym_ops:
                new_site = PeriodicSite(equiv_site._species, sym_op.operate(equiv_site.frac_coords), equiv_site._lattice)
                if new_site.is_periodic_image(this_site, tolerance=tolerance):
                    sym_trafo = sym_op
                    d_this_site2 = this_site.frac_coords - new_site.frac_coords
                    found = True
                    break
        if found:
            break
    if not found:
        raise EquivalentSiteSearchError(psite)
    return (self.structure_environments.sites_map[isite], dequivsite, dthis_site + d_this_site2, sym_trafo)