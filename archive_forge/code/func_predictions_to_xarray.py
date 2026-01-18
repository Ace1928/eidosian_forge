import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires(['predictions', f'{WARMUP_TAG}predictions'])
def predictions_to_xarray(self):
    """Convert predictions to xarray."""
    data = self._init_dict('predictions')
    data_warmup = self._init_dict(f'{WARMUP_TAG}predictions')
    if not isinstance(data, dict):
        raise TypeError('DictConverter.predictions is not a dictionary')
    if not isinstance(data_warmup, dict):
        raise TypeError('DictConverter.warmup_predictions is not a dictionary')
    predictions_attrs = self._kwargs.get('predictions_attrs')
    predictions_warmup_attrs = self._kwargs.get('predictions_warmup_attrs')
    return (dict_to_dataset(data, library=None, coords=self.coords, dims=self.pred_dims, attrs=predictions_attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=None, coords=self.coords, dims=self.pred_dims, attrs=predictions_warmup_attrs, index_origin=self.index_origin))