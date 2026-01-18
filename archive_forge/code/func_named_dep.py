from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def named_dep(self, name):
    return self.yout[..., self.odesys.names.index(name)]