from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_mft_temperature(self, j_avg):
    """
        Crude mean field estimate of critical temperature based on <J> for
        one sublattice, or solving the coupled equations for a multi-sublattice
        material.

        Args:
            j_avg (float): j_avg (float): Average exchange parameter (meV/atom)

        Returns:
            mft_t (float): Critical temperature (K)
        """
    n_sub_lattices = len(self.unique_site_ids)
    k_boltzmann = 0.0861733
    if n_sub_lattices == 1:
        mft_t = 2 * abs(j_avg) / 3 / k_boltzmann
    else:
        omega = np.zeros((n_sub_lattices, n_sub_lattices))
        ex_params = self.ex_params
        ex_params = {k: v for k, v in ex_params.items() if k != 'E0'}
        for k in ex_params:
            sites = k.split('-')
            sites = [int(num) for num in sites[:2]]
            i, j = (sites[0], sites[1])
            omega[i, j] += ex_params[k]
            omega[j, i] += ex_params[k]
        omega = omega * 2 / 3 / k_boltzmann
        eigen_vals, _eigen_vecs = np.linalg.eig(omega)
        mft_t = max(eigen_vals)
    if mft_t > 1500:
        logging.warning('This mean field estimate is too high! Probably the true low energy orderings were not given as inputs.')
    return mft_t