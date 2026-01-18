import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
@staticmethod
def get_random_sample(rng, n_unique):
    unique_times = rng.random(n_unique)
    repeats = rng.integers(1, 4, n_unique).astype(np.int32)
    times = rng.permuted(np.repeat(unique_times, repeats))
    censored = rng.random(size=times.size) > rng.random()
    sample = stats.CensoredData.right_censored(times, censored)
    return (sample, times, censored)