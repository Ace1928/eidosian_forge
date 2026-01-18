import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def create_multidimensional_model(seed=10, transpose=False):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {'y': np.random.normal(size=(ndim1, ndim2)), 'sigma': np.random.normal(size=(ndim1, ndim2))}
    posterior = {'mu': np.random.randn(nchains, ndraws), 'tau': abs(np.random.randn(nchains, ndraws)), 'eta': np.random.randn(nchains, ndraws, ndim1, ndim2), 'theta': np.random.randn(nchains, ndraws, ndim1, ndim2)}
    posterior_predictive = {'y': np.random.randn(nchains, ndraws, ndim1, ndim2)}
    sample_stats = {'energy': np.random.randn(nchains, ndraws), 'diverging': np.random.randn(nchains, ndraws) > 0.9}
    log_likelihood = {'y': np.random.randn(nchains, ndraws, ndim1, ndim2)}
    prior = {'mu': np.random.randn(nchains, ndraws) / 2, 'tau': abs(np.random.randn(nchains, ndraws)) / 2, 'eta': np.random.randn(nchains, ndraws, ndim1, ndim2) / 2, 'theta': np.random.randn(nchains, ndraws, ndim1, ndim2) / 2}
    prior_predictive = {'y': np.random.randn(nchains, ndraws, ndim1, ndim2) / 2}
    sample_stats_prior = {'energy': np.random.randn(nchains, ndraws), 'diverging': (np.random.randn(nchains, ndraws) > 0.95).astype(int)}
    model = from_dict(posterior=posterior, posterior_predictive=posterior_predictive, sample_stats=sample_stats, log_likelihood=log_likelihood, prior=prior, prior_predictive=prior_predictive, sample_stats_prior=sample_stats_prior, observed_data={'y': data['y']}, dims={'y': ['dim1', 'dim2'], 'log_likelihood': ['dim1', 'dim2']}, coords={'dim1': range(ndim1), 'dim2': range(ndim2)})
    if transpose:
        for group in model._groups:
            group_dataset = getattr(model, group)
            if all((dim in group_dataset.dims for dim in ('draw', 'chain'))):
                setattr(model, group, group_dataset.transpose(*['draw', 'chain'], ...))
    return model