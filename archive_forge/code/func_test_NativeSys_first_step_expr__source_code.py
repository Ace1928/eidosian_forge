from __future__ import (absolute_import, division, print_function)
import numpy as np
import pytest
from pyodesys.util import requires, pycvodes_double, pycvodes_klu
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem
from ._tests import (
from ._test_robertson_native import _test_chained_multi_native
from ..cvode import NativeCvodeSys as NativeSys
from pyodesys.tests.test_symbolic import _test_chained_parameter_variation
@requires('pycvodes')
def test_NativeSys_first_step_expr__source_code():
    _test_NativeSys__first_step_cb_source_code(NativeSys, -30, True, return_on_error=True, atol=1e-08, rtol=1e-08)
    _test_NativeSys__first_step_cb_source_code(NativeSys, -10, False, return_on_error=True, atol=1e-08, rtol=1e-08)