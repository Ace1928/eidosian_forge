import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('config', JUMP_TEST_DATA)
def test_jumped(config):
    seed = config['seed']
    steps = config['steps']
    mt19937 = MT19937(seed)
    mt19937.random_raw(steps)
    key = mt19937.state['state']['key']
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert mt19937.state['state']['pos'] == config['initial']['pos']
    assert sha256.hexdigest() == config['initial']['key_sha256']
    jumped = mt19937.jumped()
    key = jumped.state['state']['key']
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert jumped.state['state']['pos'] == config['jumped']['pos']
    assert sha256.hexdigest() == config['jumped']['key_sha256']