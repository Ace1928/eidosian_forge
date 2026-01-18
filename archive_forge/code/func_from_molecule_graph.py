from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
@classmethod
def from_molecule_graph(cls, mol: MoleculeGraph) -> Self:
    """
        Read a molecule from a pymatgen MoleculeGraph object.

        Args:
            mol: pymatgen MoleculeGraph object.

        Returns:
            BabelMolAdaptor object
        """
    return cls(mol.molecule)