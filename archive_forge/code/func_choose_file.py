import os
import numpy as np
from ase.gui.i18n import _
from ase import Atoms
import ase.gui.ui as ui
from ase.data import atomic_numbers, chemical_symbols
def choose_file():
    chooser = ui.ASEFileChooser(self.win.win)
    filename = chooser.go()
    if filename is None:
        return
    self.combobox.value = filename
    self.readfile(filename, format=chooser.format)