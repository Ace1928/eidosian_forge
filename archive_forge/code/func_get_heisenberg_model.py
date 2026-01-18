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
def get_heisenberg_model(self):
    """Save results of mapping to a HeisenbergModel object.

        Returns:
            HeisenbergModel: MSONable object.
        """
    hm_formula = str(self.ordered_structures_[0].reduced_formula)
    hm_structures = self.ordered_structures
    hm_energies = self.energies
    hm_cutoff = self.cutoff
    hm_tol = self.tol
    hm_sgraphs = self.sgraphs
    hm_usi = self.unique_site_ids
    hm_wids = self.wyckoff_ids
    hm_nni = self.nn_interactions
    hm_d = self.dists
    hm_em = self.ex_mat.to_json()
    hm_ep = self.get_exchange()
    hm_javg = self.estimate_exchange()
    hm_igraph = self.get_interaction_graph()
    return HeisenbergModel(hm_formula, hm_structures, hm_energies, hm_cutoff, hm_tol, hm_sgraphs, hm_usi, hm_wids, hm_nni, hm_d, hm_em, hm_ep, hm_javg, hm_igraph)