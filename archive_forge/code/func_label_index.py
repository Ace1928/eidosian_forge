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
def label_index(self):
    """
        Returns:
            The correspondence between numbers and kpoint symbols for the
        combined kpath generated when path_type = 'all'. None otherwise.
        """
    return self._label_index