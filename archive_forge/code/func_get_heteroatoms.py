from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def get_heteroatoms(self, elements=None):
    """
        Identify non-H, non-C atoms in the MoleculeGraph, returning a list of
        their node indices.

        Args:
            elements: List of elements to identify (if only certain
            functional groups are of interest).

        Returns:
            set of ints representing node indices
        """
    hetero_atoms = set()
    for node in self.molgraph.graph.nodes():
        if elements is not None:
            if str(self.species[node]) in elements:
                hetero_atoms.add(node)
        elif str(self.species[node]) not in ['C', 'H']:
            hetero_atoms.add(node)
    return hetero_atoms