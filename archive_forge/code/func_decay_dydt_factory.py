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
def decay_dydt_factory(k, names=None):
    ny = len(k) + 1

    def dydt(t, y):
        exprs = []
        for idx in range(ny):
            expr = 0
            curr_key = idx
            prev_key = idx - 1
            if names is not None:
                curr_key = names[curr_key]
                prev_key = names[prev_key]
            if idx < ny - 1:
                expr -= y[curr_key] * k[curr_key]
            if idx > 0:
                expr += y[prev_key] * k[prev_key]
            exprs.append(expr)
        if names is None:
            return exprs
        else:
            return dict(zip(names, exprs))
    return dydt