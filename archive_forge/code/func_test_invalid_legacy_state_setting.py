import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_invalid_legacy_state_setting(self):
    state = self.random_state.get_state()
    new_state = ('Unknown',) + state[1:]
    assert_raises(ValueError, self.random_state.set_state, new_state)
    assert_raises(TypeError, self.random_state.set_state, np.array(new_state, dtype=object))
    state = self.random_state.get_state(legacy=False)
    del state['bit_generator']
    assert_raises(ValueError, self.random_state.set_state, state)