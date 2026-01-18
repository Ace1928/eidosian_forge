import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_additivechi2sampler_get_feature_names_out():
    """Check get_feature_names_out for AdditiveChi2Sampler."""
    rng = np.random.RandomState(0)
    X = rng.random_sample(size=(300, 3))
    chi2_sampler = AdditiveChi2Sampler(sample_steps=3).fit(X)
    input_names = ['f0', 'f1', 'f2']
    suffixes = ['f0_sqrt', 'f1_sqrt', 'f2_sqrt', 'f0_cos1', 'f1_cos1', 'f2_cos1', 'f0_sin1', 'f1_sin1', 'f2_sin1', 'f0_cos2', 'f1_cos2', 'f2_cos2', 'f0_sin2', 'f1_sin2', 'f2_sin2']
    names_out = chi2_sampler.get_feature_names_out(input_features=input_names)
    expected_names = [f'additivechi2sampler_{suffix}' for suffix in suffixes]
    assert_array_equal(names_out, expected_names)