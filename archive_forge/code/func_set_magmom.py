from functools import partial
from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.gui.widgets import Element
from ase.gui.utils import get_magmoms
def set_magmom(self):
    magmoms = get_magmoms(self.gui.atoms)
    magmoms[self.selection()] = self.magmom.value
    self.gui.atoms.set_initial_magnetic_moments(magmoms)
    self.gui.draw()