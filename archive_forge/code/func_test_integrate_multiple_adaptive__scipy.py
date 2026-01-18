from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('scipy')
def test_integrate_multiple_adaptive__scipy():
    _test_integrate_multiple_adaptive(ODESys(sine, sine_jac), integrator='scipy', method='bdf', name='vode', first_step=1e-09)