from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_center(self, *args):
    atoms = self.gui.atoms.copy()
    atoms.center()
    self.gui.new_atoms(atoms)