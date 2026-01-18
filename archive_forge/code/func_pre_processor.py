from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def pre_processor(x, p):
    return (np.log(np.asarray(x) + math.exp(small)), p)