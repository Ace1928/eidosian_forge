from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
def set_group_info(self, group_name, group_number, insertion_code, group_type, atom_count, bond_count, single_letter_code, sequence_index, secondary_structure_type):
    """Set the information for a group.

        :param group_name: the name of this group, e.g. LYS
        :param group_number: the residue number of this group
        :param insertion_code: the insertion code for this group
        :param group_type: a string indicating the type of group (as found in the chemcomp dictionary.
            Empty string if none available.
        :param atom_count: the number of atoms in the group
        :param bond_count: the number of unique bonds in the group
        :param single_letter_code: the single letter code of the group
        :param sequence_index: the index of this group in the sequence defined by the entity
        :param secondary_structure_type: the type of secondary structure used
            (types are according to DSSP and number to type mappings are defined in the specification)

        """
    if insertion_code == '\x00':
        insertion_code = ' '
    self.structure_builder.init_seg(' ')
    self.structure_builder.init_residue(group_name, self.this_type, group_number, insertion_code)