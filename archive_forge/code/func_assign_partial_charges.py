from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule
def assign_partial_charges(openff_mol: tk.Molecule, atom_map: dict[int, int], charge_method: str, partial_charges: None | list[float]) -> tk.Molecule:
    """
    Assign partial charges to an OpenFF Molecule.

    If partial charges are provided, assigns them to the molecule
    based on the atom mapping. If the molecule has only one atom,
    assigns the total charge as the partial charge. Otherwise,
    assigns partial charges using the specified charge method.

    Args:
        openff_mol (tk.Molecule): The OpenFF Molecule to assign partial charges to.
        atom_map (Dict[int, int]): A dictionary representing the atom mapping.
        charge_method (str): The charge method to use if partial charges are
            not provided.
        partial_charges (Union[None, List[float]]): A list of partial charges to
            assign or None to use the charge method.

    Returns:
        tk.Molecule: The OpenFF Molecule with assigned partial charges.
    """
    if partial_charges is not None:
        partial_charges = np.array(partial_charges)
        chargs = partial_charges[list(atom_map.values())]
        openff_mol.partial_charges = chargs * unit.elementary_charge
    elif openff_mol.n_atoms == 1:
        openff_mol.partial_charges = np.array([openff_mol.total_charge.magnitude]) * unit.elementary_charge
    else:
        openff_mol.assign_partial_charges(charge_method)
    return openff_mol