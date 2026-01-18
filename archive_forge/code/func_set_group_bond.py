from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_group_bond(self, atom_index_one, atom_index_two, bond_order):
    """Add bonds within a group.

        :param atom_index_one: the integer atom index (in the group) of the first partner in the bond
        :param atom_index_two: the integer atom index (in the group) of the second partner in the bond
        :param bond_order: the integer bond order

        """