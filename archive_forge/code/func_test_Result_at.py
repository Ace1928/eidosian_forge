from __future__ import (absolute_import, division, print_function)
import numpy as np
from .. import ODESys
from ..util import requires
from .test_core import sine, sine_jac
@requires('scipy')
def test_Result_at():
    _test_sine(use_deriv=False)