from copy import copy
from ase.gui.i18n import _
import numpy as np
import ase
import ase.data
import ase.gui.ui as ui
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.cluster.hexagonal import HexagonalClosedPacked, Graphite
from ase.cluster import wulff_construction
from ase.gui.widgets import Element, pybutton
import ase
import ase
from ase.cluster import wulff_construction
def makeatoms(self, *args):
    """Make the atoms according to the current specification."""
    symbol = self.element.symbol
    if symbol is None:
        self.clearatoms()
        self.makeinfo()
        return False
    struct = self.structure.value
    if self.needs_2lat[struct]:
        lc = {'a': self.a.value, 'c': self.c.value}
        lc_str = str(lc)
    else:
        lc = self.a.value
        lc_str = '%.5f' % (lc,)
    if self.method.value == 'wulff':
        surfaces = [x[0] for x in self.direction_table]
        surfaceenergies = [x[1].value for x in self.direction_table_rows.rows]
        self.update_size_diameter(update=False)
        rounding = self.round_radio.value
        self.atoms = wulff_construction(symbol, surfaces, surfaceenergies, self.size_natoms.value, self.factory[struct], rounding, lc)
        python = py_template_wulff % {'element': symbol, 'surfaces': str(surfaces), 'energies': str(surfaceenergies), 'latconst': lc_str, 'natoms': self.size_natoms.value, 'structure': struct, 'rounding': rounding}
    else:
        surfaces = [x[0] for x in self.direction_table]
        layers = [x[1].value for x in self.direction_table_rows.rows]
        self.atoms = self.factory[struct](symbol, copy(surfaces), layers, latticeconstant=lc)
        imp = self.import_names[struct]
        python = py_template_layers % {'import': imp, 'element': symbol, 'surfaces': str(surfaces), 'layers': str(layers), 'latconst': lc_str, 'factory': imp.split()[-1]}
    self.makeinfo()
    return python