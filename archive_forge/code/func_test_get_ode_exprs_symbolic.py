from __future__ import (absolute_import, division, print_function)
import numpy as np
from ..util import import_
import pytest
from .. import ODESys
from ..core import integrate_chained
from ..symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys
from ..util import requires, pycvodes_double
from ._robertson import run_integration, get_ode_exprs
@requires('sym', 'sympy', 'pycvodes')
@pycvodes_double
def test_get_ode_exprs_symbolic():
    _test_goe(symbolic=True, logc=True, logt=False, zero_conc=1e-20, atol=1e-08, rtol=1e-10, extra_forgive=2, first_step=1e-14)
    _test_goe(symbolic=True, logc=True, logt=True, zero_conc=1e-20, zero_time=1e-12, atol=1e-08, rtol=1e-12, extra_forgive=2)
    _test_goe(symbolic=True, logc=False, logt=True, zero_conc=0, zero_time=1e-12, atol=1e-09, rtol=5e-13, extra_forgive=0.4)
    for reduced in range(4):
        _test_goe(symbolic=True, reduced=reduced, first_step=1e-14, extra_forgive=5)
        if reduced != 2:
            _test_goe(symbolic=True, reduced=reduced, logc=True, logt=False, zero_conc=1e-16, atol=1e-08, rtol=1e-10, extra_forgive=2, first_step=1e-14)
        if reduced == 3:
            _test_goe(symbolic=True, reduced=reduced, logc=True, logt=True, zero_conc=1e-18, zero_time=1e-12, atol=1e-12, rtol=1e-10, extra_forgive=0.0002)
        if reduced != 3:
            _test_goe(symbolic=True, reduced=reduced, logc=False, logt=True, zero_time=1e-12, atol=1e-12, rtol=5e-13, extra_forgive=1, first_step=1e-14)
            _test_goe(symbolic=True, reduced=reduced, logc=False, logt=True, zero_time=1e-09, atol=1e-13, rtol=1e-14, first_step=1e-10, extra_forgive=2)