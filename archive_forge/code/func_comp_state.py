import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    elif type(state1) != type(state2):
        identical &= type(state1) == type(state2)
    elif isinstance(state1, (list, tuple, np.ndarray)) and isinstance(state2, (list, tuple, np.ndarray)):
        for s1, s2 in zip(state1, state2):
            identical &= comp_state(s1, s2)
    else:
        identical &= state1 == state2
    return identical