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
def get_charge_transfer(self, atom_index, charge_type='ddec'):
    """Returns the charge transferred for a particular atom. A positive value means
        that the site has gained electron density (i.e. exhibits anionic character)
        whereas a negative value means the site has lost electron density (i.e. exhibits
        cationic character). This is the same thing as the negative of the partial atomic
        charge.

        Args:
            atom_index (int): Index of atom to get charge transfer for.
            charge_type (str): Type of charge to use ("ddec" or "cm5").

        Returns:
            float: charge transferred at atom_index
        """
    if charge_type.lower() not in ['ddec', 'cm5']:
        raise ValueError(f'Invalid charge_type={charge_type!r}')
    if charge_type.lower() == 'ddec':
        charge_transfer = -self.ddec_charges[atom_index]
    elif charge_type.lower() == 'cm5':
        charge_transfer = -self.cm5_charges[atom_index]
    return charge_transfer