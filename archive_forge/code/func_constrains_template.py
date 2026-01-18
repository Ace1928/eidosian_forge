from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
from monty.json import MSONable
@staticmethod
def constrains_template(molecule, reference_fnm, constraints) -> str:
    """
        Args:
            molecule (Molecule): Molecule the constraints will be performed on
            reference_fnm (str): Name of file containing reference structure in same directory
            constraints (dict): Dictionary of common editable parameters for .constrains file.
                {"atoms": [List of 1-indexed atoms to fix], "force_constant": float]

        Returns:
            str: for .constrains file
        """
    atoms_to_constrain = constraints['atoms']
    force_constant = constraints['force_constant']
    mol = molecule
    atoms_for_mtd = [idx for idx in range(1, len(mol) + 1) if idx not in atoms_to_constrain]
    interval_list = [atoms_for_mtd[0]]
    for idx, val in enumerate(atoms_for_mtd, start=1):
        if val + 1 not in atoms_for_mtd:
            interval_list.append(val)
            if idx != len(atoms_for_mtd):
                interval_list.append(atoms_for_mtd[idx])
    allowed_mtd_string = ','.join([f'{interval_list[i]}-{interval_list[i + 1]}' for i in range(len(interval_list)) if i % 2 == 0])
    return f'$constrain\n  atoms: {','.join(map(str, atoms_to_constrain))}\n  force constant={force_constant}\n  reference={reference_fnm}\n$metadyn\n  atoms: {allowed_mtd_string}\n$end'