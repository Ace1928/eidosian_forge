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
def get_atomic_volume(self):
    s = self.structure.value
    a = self.a.value
    c = self.c.value
    if s == 'fcc':
        return a ** 3 / 4
    elif s == 'bcc':
        return a ** 3 / 2
    elif s == 'sc':
        return a ** 3
    elif s == 'hcp':
        return np.sqrt(3.0) / 2 * a * a * c / 2
    elif s == 'graphite':
        return np.sqrt(3.0) / 2 * a * a * c / 4