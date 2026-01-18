from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule
def mol_graph_to_openff_mol(mol_graph: MoleculeGraph) -> tk.Molecule:
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Molecule.

    Args:
        mol_graph (MoleculeGraph): The Pymatgen MoleculeGraph to be converted.

    Returns:
        tk.Molecule: The converted OpenFF Molecule.
    """
    p_table = {str(el): el.Z for el in Element}
    openff_mol = tk.Molecule()
    partial_charges = []
    for i_node in range(len(mol_graph.graph.nodes)):
        node = mol_graph.graph.nodes[i_node]
        atomic_number = node.get('atomic_number') or p_table[mol_graph.molecule[i_node].species_string]
        formal_charge = node.get('formal_charge')
        if formal_charge is None:
            formal_charge = (i_node == 0) * int(round(mol_graph.molecule.charge, 0)) * unit.elementary_charge
        is_aromatic = node.get('is_aromatic') or False
        openff_mol.add_atom(atomic_number, formal_charge, is_aromatic=is_aromatic)
        partial_charge = node.get('partial_charge')
        if isinstance(partial_charge, Quantity):
            partial_charge = partial_charge.magnitude
        partial_charges.append(partial_charge)
    charge_array = np.array(partial_charges)
    if np.not_equal(charge_array, None).all():
        openff_mol.partial_charges = charge_array * unit.elementary_charge
    for i_node, j, bond_data in mol_graph.graph.edges(data=True):
        bond_order = bond_data.get('bond_order') or 1
        is_aromatic = bond_data.get('is_aromatic') or False
        openff_mol.add_bond(i_node, j, bond_order, is_aromatic=is_aromatic)
    openff_mol.add_conformer(mol_graph.molecule.cart_coords * unit.angstrom)
    return openff_mol