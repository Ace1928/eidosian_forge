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
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('method,forgive', zip('adams bdf'.split(), (2.4, 5.0)))
def test_cvode_predefined(method, forgive):
    _cvode([10 ** i for i in range(-15, 1)], method, forgive)