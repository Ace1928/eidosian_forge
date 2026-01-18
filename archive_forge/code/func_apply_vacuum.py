from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_vacuum(self, *args):
    atoms = self.gui.atoms.copy()
    axis = []
    for index, pbc in enumerate(atoms.pbc):
        if not pbc:
            axis.append(index)
    atoms.center(vacuum=self.vacuum.value, axis=axis)
    self.gui.new_atoms(atoms)