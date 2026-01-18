from ase.gui.i18n import _, ngettext
import ase.gui.ui as ui
import ase.build as build
from ase.data import reference_states
from ase.gui.widgets import Element, pybutton
from ase.build import {func}
def get_lattice(self, *args):
    if self.element.symbol is None:
        return
    ref = reference_states[self.element.Z]
    symmetry = 'unknown'
    for struct in surfaces:
        if struct[0] == self.structure.value:
            symmetry = struct[1]
    if ref['symmetry'] != symmetry:
        self.structure_warn.text = _('Error: Reference values assume {} crystal structure for {}!').format(ref['symmetry'], self.element.symbol)
    elif symmetry == 'fcc' or symmetry == 'bcc' or symmetry == 'diamond':
        self.lattice_a.value = ref['a']
    elif symmetry == 'hcp':
        self.lattice_a.value = ref['a']
        self.lattice_c.value = ref['a'] * ref['c/a']
    self.make()