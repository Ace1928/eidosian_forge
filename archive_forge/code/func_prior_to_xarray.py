import typing as tp
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import xarray
from .inference_data import InferenceData
from ..rcparams import rcParams
from .base import dict_to_dataset
def prior_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
    """Extract posterior samples from fit."""
    if self.prior is None:
        return None
    return self._pyjags_samples_to_xarray(self.prior)