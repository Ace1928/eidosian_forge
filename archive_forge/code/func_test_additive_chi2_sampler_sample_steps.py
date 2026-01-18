import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('method', ['fit', 'fit_transform', 'transform'])
@pytest.mark.parametrize('sample_steps', range(1, 4))
def test_additive_chi2_sampler_sample_steps(method, sample_steps):
    """Check that the input sample step doesn't raise an error
    and that sample interval doesn't change after fit.
    """
    transformer = AdditiveChi2Sampler(sample_steps=sample_steps)
    getattr(transformer, method)(X)
    sample_interval = 0.5
    transformer = AdditiveChi2Sampler(sample_steps=sample_steps, sample_interval=sample_interval)
    getattr(transformer, method)(X)
    assert transformer.sample_interval == sample_interval