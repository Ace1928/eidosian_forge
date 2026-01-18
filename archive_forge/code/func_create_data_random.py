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
def create_data_random(groups=None, seed=10):
    """Create InferenceData object using random data."""
    if groups is None:
        groups = ['posterior', 'sample_stats', 'observed_data', 'posterior_predictive']
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(4, 500, 8))
    idata_dict = dict(posterior={'a': data[..., 0], 'b': data}, sample_stats={'a': data[..., 0], 'b': data}, observed_data={'b': data[0, 0, :]}, posterior_predictive={'a': data[..., 0], 'b': data}, prior={'a': data[..., 0], 'b': data}, prior_predictive={'a': data[..., 0], 'b': data}, warmup_posterior={'a': data[..., 0], 'b': data}, warmup_posterior_predictive={'a': data[..., 0], 'b': data}, warmup_prior={'a': data[..., 0], 'b': data})
    idata = from_dict(**{group: ary for group, ary in idata_dict.items() if group in groups}, save_warmup=True)
    return idata