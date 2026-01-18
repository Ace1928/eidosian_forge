from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
def analytic(tout, init_y, p):
    y0ref = init_y['a'] * np.exp(-tout ** (p['e'] + 1) / (p['e'] + 1))
    return np.array([y0ref, init_y['a'] - y0ref + init_y['b']]).T