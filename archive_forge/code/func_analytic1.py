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
def analytic1(i, p, a):
    assert i > 0
    assert p >= 0
    assert a > 0
    from scipy.special import binom
    return binom(p + i - 1, p) * a ** (i - 1) * (a + 1) ** (-i - p)