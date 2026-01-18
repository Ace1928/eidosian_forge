from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def link_marked_atoms(self, atoms):
    """
        Take a list of marked "interesting" atoms (heteroatoms, special carbons)
        and attempt to connect them, returning a list of disjoint groups of
        special atoms (and their connected hydrogens).

        Args:
            atoms: set of marked "interesting" atoms, presumably identified
                using other functions in this class.

        Returns:
            list of sets of ints, representing groups of connected atoms
        """
    hydrogens = {n for n in self.molgraph.graph.nodes if str(self.species[n]) == 'H'}
    subgraph = self.molgraph.graph.subgraph(list(atoms)).to_undirected()
    func_groups = []
    for func_grp in nx.connected_components(subgraph):
        grp_hs = set()
        for node in func_grp:
            neighbors = self.molgraph.graph[node]
            for neighbor in neighbors:
                if neighbor in hydrogens:
                    grp_hs.add(neighbor)
        func_grp = func_grp | grp_hs
        func_groups.append(func_grp)
    return func_groups