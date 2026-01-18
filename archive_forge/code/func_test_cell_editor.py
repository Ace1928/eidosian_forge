import os
from pathlib import Path
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule, bulk
import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.gui import GUI
from ase.gui.save import save_dialog
from ase.gui.quickinfo import info
def test_cell_editor(gui):
    au = bulk('Au')
    gui.new_atoms(au.copy())
    dia = gui.cell_editor()
    ti = bulk('Ti')
    dia.update(ti.cell, ti.pbc)
    dia.apply_vectors()
    tol = 3e-07
    assert np.abs(gui.atoms.cell - ti.cell).max() < tol
    dia.update(ti.cell * 2, ti.pbc)
    dia.apply_magnitudes()
    assert np.abs(gui.atoms.cell - 2 * ti.cell).max() < tol
    dia.update(np.eye(3), ti.pbc)
    dia.apply_angles()
    assert abs(gui.atoms.cell.angles() - 90).max() < tol
    newpbc = [0, 1, 0]
    dia.update(np.eye(3), newpbc)
    dia.apply_pbc()
    assert (gui.atoms.pbc == newpbc).all()