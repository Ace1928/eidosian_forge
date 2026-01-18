from math import sqrt
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.gui.defaults import read_defaults
from ase.io import read, write, string2index
from ase.gui.i18n import _
from ase.geometry import find_mic
import warnings
def repeat_images(self, repeat):
    from ase.constraints import FixAtoms
    repeat = np.array(repeat)
    oldprod = self.repeat.prod()
    images = []
    constraints_removed = False
    for i, atoms in enumerate(self):
        refcell = atoms.get_cell()
        fa = []
        for c in atoms._constraints:
            if isinstance(c, FixAtoms):
                fa.append(c)
            else:
                constraints_removed = True
        atoms.set_constraint(fa)
        results = self.repeat_results(atoms, repeat, oldprod)
        del atoms[len(atoms) // oldprod:]
        atoms *= repeat
        atoms.cell = refcell
        atoms.calc = SinglePointCalculator(atoms, **results)
        images.append(atoms)
    if constraints_removed:
        from ase.gui.ui import tk, showwarning
        tmpwindow = tk.Tk()
        tmpwindow.withdraw()
        showwarning(_('Constraints discarded'), _('Constraints other than FixAtoms have been discarded.'))
        tmpwindow.destroy()
    self.initialize(images, filenames=self.filenames)
    self.repeat = repeat