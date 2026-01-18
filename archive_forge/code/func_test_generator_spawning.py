import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_generator_spawning():
    """ Test spawning new generators and bit_generators directly.
    """
    rng = np.random.default_rng()
    seq = rng.bit_generator.seed_seq
    new_ss = seq.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5)]
    assert [c.spawn_key for c in new_ss] == expected_keys
    new_bgs = rng.bit_generator.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5, 10)]
    assert [bg.seed_seq.spawn_key for bg in new_bgs] == expected_keys
    new_rngs = rng.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(10, 15)]
    found_keys = [rng.bit_generator.seed_seq.spawn_key for rng in new_rngs]
    assert found_keys == expected_keys
    assert new_rngs[0].uniform() != new_rngs[1].uniform()