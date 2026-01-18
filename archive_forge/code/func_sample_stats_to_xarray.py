import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires(['sample_stats', f'{WARMUP_TAG}sample_stats'])
def sample_stats_to_xarray(self):
    """Convert sample_stats samples to xarray."""
    data = self._init_dict('sample_stats')
    data_warmup = self._init_dict(f'{WARMUP_TAG}sample_stats')
    if not isinstance(data, dict):
        raise TypeError('DictConverter.sample_stats is not a dictionary')
    if not isinstance(data_warmup, dict):
        raise TypeError('DictConverter.warmup_sample_stats is not a dictionary')
    if 'log_likelihood' in data:
        warnings.warn('log_likelihood variable found in sample_stats. Storing log_likelihood data in sample_stats group will be deprecated in favour of storing them in the log_likelihood group.', PendingDeprecationWarning)
    sample_stats_attrs = self._kwargs.get('sample_stats_attrs')
    sample_stats_warmup_attrs = self._kwargs.get('sample_stats_warmup_attrs')
    return (dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims, attrs=sample_stats_attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=sample_stats_warmup_attrs, index_origin=self.index_origin))