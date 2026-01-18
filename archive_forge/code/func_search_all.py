import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Selection import unfold_entities, entity_levels, uniqueify
def search_all(self, radius, level='A'):
    """All neighbor search.

        Search all entities that have atoms pairs within
        radius.

        Arguments:
         - radius - float
         - level - char (A, R, C, M, S)

        """
    if level not in entity_levels:
        raise PDBException(f'{level}: Unknown level')
    neighbors = self.kdt.neighbor_search(radius)
    atom_list = self.atom_list
    atom_pair_list = []
    for neighbor in neighbors:
        i1 = neighbor.index1
        i2 = neighbor.index2
        a1 = atom_list[i1]
        a2 = atom_list[i2]
        atom_pair_list.append((a1, a2))
    if level == 'A':
        return atom_pair_list
    next_level_pair_list = atom_pair_list
    for next_level in ['R', 'C', 'M', 'S']:
        next_level_pair_list = self._get_unique_parent_pairs(next_level_pair_list)
        if level == next_level:
            return next_level_pair_list