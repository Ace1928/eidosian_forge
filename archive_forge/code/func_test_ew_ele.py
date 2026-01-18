from __future__ import (absolute_import, division, print_function)
import numpy as np
import pytest
from pyodesys.util import requires, pycvodes_double, pycvodes_klu
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem
from ._tests import (
from ._test_robertson_native import _test_chained_multi_native
from ..cvode import NativeCvodeSys as NativeSys
from pyodesys.tests.test_symbolic import _test_chained_parameter_variation
@requires('sym', 'pycvodes')
def test_ew_ele():
    for tst in [_test_multiple_predefined, _test_multiple_adaptive]:
        results = tst(NativeSys, atol=1e-10, rtol=1e-10, ew_ele=True, nsteps=1400)
        for res in results:
            ee = res.info['ew_ele']
            assert ee.ndim == 3 and ee.shape[1] == 2