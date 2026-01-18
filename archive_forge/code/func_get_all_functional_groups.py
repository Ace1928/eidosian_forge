from __future__ import annotations
import copy
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
def get_all_functional_groups(self, elements=None, func_groups=None, catch_basic=True):
    """
        Identify all functional groups (or all within a certain subset) in the
        molecule, combining the methods described above.

        Args:
            elements: List of elements that will qualify a carbon as special
                (if only certain functional groups are of interest).
                Default None.
            func_groups: List of strs representing the functional groups of
                interest. Default to None, meaning that all of the functional groups
                defined in this function will be sought.
            catch_basic: bool. If True, use get_basic_functional_groups and
                other methods

        Returns:
            list of sets of ints, representing groups of connected atoms
        """
    heteroatoms = self.get_heteroatoms(elements=elements)
    special_cs = self.get_special_carbon(elements=elements)
    groups = self.link_marked_atoms(heteroatoms | special_cs)
    if catch_basic:
        groups += self.get_basic_functional_groups(func_groups=func_groups)
    return groups