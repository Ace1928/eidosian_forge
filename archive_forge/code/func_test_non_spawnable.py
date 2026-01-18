import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_non_spawnable():
    from numpy.random.bit_generator import ISeedSequence

    class FakeSeedSequence:

        def generate_state(self, n_words, dtype=np.uint32):
            return np.zeros(n_words, dtype=dtype)
    ISeedSequence.register(FakeSeedSequence)
    rng = np.random.default_rng(FakeSeedSequence())
    with pytest.raises(TypeError, match='The underlying SeedSequence'):
        rng.spawn(5)
    with pytest.raises(TypeError, match='The underlying SeedSequence'):
        rng.bit_generator.spawn(5)