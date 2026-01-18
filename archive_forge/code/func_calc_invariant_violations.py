from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def calc_invariant_violations(self, xyp=None):
    invar = self.odesys.get_invariants_callback()
    val = invar(*(xyp or self._internals()))
    return val - val[0, :]