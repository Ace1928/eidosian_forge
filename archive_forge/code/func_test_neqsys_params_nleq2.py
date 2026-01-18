from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
@pytest.mark.skipif(not HAVE_PYNLEQ2, reason='pynleq2 not installed on system.')
def test_neqsys_params_nleq2():
    _test_neqsys_params('nleq2')