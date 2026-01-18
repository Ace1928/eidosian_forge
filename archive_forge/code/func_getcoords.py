import os
import numpy as np
from ase.gui.i18n import _
from ase import Atoms
import ase.gui.ui as ui
from ase.data import atomic_numbers, chemical_symbols
def getcoords(self):
    addcoords = np.array([spinner.value for spinner in self.spinners])
    pos = self.gui.atoms.positions
    if self.gui.images.selected[:len(pos)].any():
        pos = pos[self.gui.images.selected[:len(pos)]]
        center = pos.mean(0)
        addcoords += center
    return addcoords