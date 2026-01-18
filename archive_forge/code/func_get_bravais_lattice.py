import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
def get_bravais_lattice(self, eps=0.0002, *, pbc=True):
    """Return :class:`~ase.lattice.BravaisLattice` for this cell:

        >>> cell = Cell.fromcellpar([4, 4, 4, 60, 60, 60])
        >>> print(cell.get_bravais_lattice())
        FCC(a=5.65685)

        .. note:: The Bravais lattice object follows the AFlow
           conventions.  ``cell.get_bravais_lattice().tocell()`` may
           differ from the original cell by a permutation or other
           operation which maps it to the AFlow convention.  For
           example, the orthorhombic lattice enforces a < b < c.

           To build a bandpath for a particular cell, use
           :meth:`ase.cell.Cell.bandpath` instead of this method.
           This maps the kpoints back to the original input cell.

        """
    from ase.lattice import identify_lattice
    pbc = self.any(1) & pbc2pbc(pbc)
    lat, op = identify_lattice(self, eps=eps, pbc=pbc)
    return lat