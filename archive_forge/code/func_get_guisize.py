from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.io.pov import write_pov, get_bondpairs
from os import unlink
import numpy as np
def get_guisize(self):
    win = self.gui.window.win
    return (win.winfo_width(), win.winfo_height())