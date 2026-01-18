from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def store_dataset(self, dataset):
    """
        in stores, variables are all variables AND coordinates
        in xarray.Dataset variables are variables NOT coordinates,
        so here we pass the whole dataset in instead of doing
        dataset.variables
        """
    self.store(dataset, dataset.attrs)