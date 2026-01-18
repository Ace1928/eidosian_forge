import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def site_dos(self, atom, orbital):
    """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
    if self.resort:
        atom = self.resort[atom]
    if isinstance(orbital, int):
        return self._site_dos[atom, orbital + 1, :]
    n = self._site_dos.shape[1]
    from .vasp_data import PDOS_orbital_names_and_DOSCAR_column
    norb = PDOS_orbital_names_and_DOSCAR_column[n]
    return self._site_dos[atom, norb[orbital.lower()], :]