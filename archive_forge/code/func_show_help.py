from ase.gui.i18n import _
import ase.data
import ase.gui.ui as ui
from ase import Atoms
def show_help(self):
    msg = _('Enter a chemical symbol or the atomic number.')
    ui.showinfo(_('Info'), msg)