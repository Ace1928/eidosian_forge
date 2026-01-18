from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def get_charge(self, atom_index, nelect=None, charge_type='ddec'):
    """Convenience method to get the charge on a particular atom using the same
        sign convention as the BaderAnalysis. Note that this is *not* the partial
        atomic charge. This value is nelect (e.g. ZVAL from the POTCAR) + the
        charge transferred. If you want the partial atomic charge, use
        get_partial_charge().

        Args:
            atom_index (int): Index of atom to get charge for.
            nelect (int): number of electrons associated with an isolated atom at this index.
            For most DFT codes this corresponds to the number of valence electrons
            associated with the pseudopotential. If None, this value will be automatically
            obtained from the POTCAR (if present).
                Default: None.
            charge_type (str): Type of charge to use ("ddec" or "cm5").

        Returns:
            float: charge on atom_index
        """
    if nelect:
        charge = nelect + self.get_charge_transfer(atom_index, charge_type=charge_type)
    elif self.potcar and self.natoms:
        charge = None
        potcar_indices = []
        for idx, val in enumerate(self.natoms):
            potcar_indices += [idx] * val
        nelect = self.potcar[potcar_indices[atom_index]].nelectrons
        charge = nelect + self.get_charge_transfer(atom_index, charge_type=charge_type)
    else:
        charge = None
    return charge