from ase.gui.i18n import _, ngettext
import ase.gui.ui as ui
import ase.build as build
from ase.data import reference_states
from ase.gui.widgets import Element, pybutton
from ase.build import {func}
def structure_changed(self, *args):
    for surface in surfaces:
        if surface[0] == self.structure.value:
            if surface[2] == 'ortho':
                self.orthogonal.var.set(True)
                self.orthogonal.check['state'] = ['disabled']
            elif surface[2] == 'non-ortho':
                self.orthogonal.var.set(False)
                self.orthogonal.check['state'] = ['disabled']
            else:
                self.orthogonal.check['state'] = ['normal']
            if surface[1] == _('hcp'):
                self.lattice_c.active = True
                self.lattice_c.value = round(self.lattice_a.value * (8.0 / 3.0) ** 0.5, 3)
            else:
                self.lattice_c.active = False
                self.lattice_c.value = 'None'
    self.get_lattice()