import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_nmant():
    for t in IEEE_floats:
        assert type_info(t)['nmant'] == np.finfo(t).nmant
    if (LD_INFO['nmant'], LD_INFO['nexp']) == (63, 15):
        assert type_info(np.longdouble)['nmant'] == 63