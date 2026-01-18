from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.util.due import Doi, due
@staticmethod
def get_13_bonds(priority_bonds):
    """
        Args:
            priority_bonds ():

        Returns:
            tuple: 13 bonds
        """
    all_bond_pairs = list(itertools.combinations(priority_bonds, r=2))
    all_2_bond_atoms = [set(b1 + b2) for b1, b2 in all_bond_pairs]
    all_13_bond_atoms = [a for a in all_2_bond_atoms if len(a) == 3]
    all_2_and_13_bonds = {tuple(sorted(b)) for b in itertools.chain(*(itertools.combinations(p, 2) for p in all_13_bond_atoms))}
    bonds_13 = all_2_and_13_bonds - {tuple(b) for b in priority_bonds}
    return tuple(sorted(bonds_13))