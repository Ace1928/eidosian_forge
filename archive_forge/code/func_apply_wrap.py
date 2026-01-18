from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_wrap(self, *args):
    atoms = self.gui.atoms.copy()
    atoms.wrap()
    self.gui.new_atoms(atoms)