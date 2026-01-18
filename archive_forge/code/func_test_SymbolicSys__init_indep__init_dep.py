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
@requires('sym')
def test_SymbolicSys__init_indep__init_dep():
    odesys = SymbolicSys.from_callback(lambda x, y, p, be: [-y[0], y[0]], 2, names=['foo', 'bar'], indep_name='t', init_indep=True, init_dep=True)
    assert odesys.init_indep.name == 'i_t'
    assert [dep.name for dep in odesys.init_dep] == ['i_foo', 'i_bar']