from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_ChainedNeqSys():
    neqsys_log = _get_cneqsys3(-60)
    neqsys_lin = _get_cneqsys2()
    chained = ChainedNeqSys([neqsys_log, neqsys_lin])
    info_dicts = _check_NaCl(chained, [None], 2, method='lm')
    for nfo in info_dicts:
        assert nfo['intermediate_info'][0]['success'] and nfo['intermediate_info'][1]['success']