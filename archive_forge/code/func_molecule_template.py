from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def molecule_template(molecule: Molecule | list[Molecule] | Literal['read']) -> str:
    """
        Args:
            molecule (Molecule, list of Molecules, or "read").

        Returns:
            str: Molecule template.
        """
    mol_list = []
    mol_list.append('$molecule')
    if isinstance(molecule, list) and len(molecule) == 1:
        molecule = molecule[0]
    if isinstance(molecule, str):
        if molecule == 'read':
            mol_list.append(' read')
        else:
            raise ValueError('The only acceptable text value for molecule is "read"')
    elif isinstance(molecule, Molecule):
        mol_list.append(f' {int(molecule.charge)} {molecule.spin_multiplicity}')
        for site in molecule:
            mol_list.append(f' {site.species_string}     {site.x: .10f}     {site.y: .10f}     {site.z: .10f}')
    else:
        overall_charge = sum((x.charge for x in molecule))
        unpaired_electrons = sum((x.spin_multiplicity - 1 for x in molecule))
        overall_spin = unpaired_electrons + 1
        mol_list.append(f' {int(overall_charge)} {int(overall_spin)}')
        for fragment in molecule:
            mol_list.extend(('--', f' {int(fragment.charge)} {fragment.spin_multiplicity}'))
            for site in fragment:
                mol_list.append(f' {site.species_string}     {site.x: .10f}     {site.y: .10f}     {site.z: .10f}')
    mol_list.append('$end')
    return '\n'.join(mol_list)