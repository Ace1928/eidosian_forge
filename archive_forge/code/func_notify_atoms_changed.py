from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def notify_atoms_changed(self):
    atoms = self.gui.atoms
    self.update(atoms.cell, atoms.pbc)