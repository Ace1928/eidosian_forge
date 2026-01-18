from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def apply_pbc(self, *args):
    atoms = self.gui.atoms.copy()
    pbc = [pbc.var.get() for pbc in self.pbc]
    atoms.set_pbc(pbc)
    self.gui.new_atoms(atoms)