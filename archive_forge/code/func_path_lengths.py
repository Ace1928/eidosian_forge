from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
@property
def path_lengths(self):
    """
        Returns:
            List of lengths of the Latimer and Munro, Setyawan and Curtarolo, and Hinuma
            conventions in the combined HighSymmKpath object when path_type = 'all' respectively.
            None otherwise.
        """
    return self._path_lengths