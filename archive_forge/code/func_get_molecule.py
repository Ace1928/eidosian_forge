from __future__ import annotations
import warnings
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.core.structure import Molecule, Structure
@staticmethod
def get_molecule(atoms: Atoms, cls: type[Molecule]=Molecule, **cls_kwargs) -> Molecule:
    """
        Returns pymatgen molecule from ASE Atoms.

        Args:
            atoms: ASE Atoms object
            cls: The Molecule class to instantiate (defaults to pymatgen molecule)
            **cls_kwargs: Any additional kwargs to pass to the cls

        Returns:
            Molecule: Equivalent pymatgen.core.structure.Molecule
        """
    molecule = AseAtomsAdaptor.get_structure(atoms, cls=cls, **cls_kwargs)
    try:
        charge = atoms.charge
    except AttributeError:
        charge = round(np.sum(atoms.get_initial_charges())) if atoms.has('initial_charges') else 0
    try:
        spin_mult = atoms.spin_multiplicity
    except AttributeError:
        spin_mult = round(np.sum(atoms.get_initial_magnetic_moments())) + 1 if atoms.has('initial_magmoms') else 1
    molecule.set_charge_and_spin(charge, spin_multiplicity=spin_mult)
    return molecule