from __future__ import annotations
from importlib.metadata import EntryPoint
from typing import Any
import numpy as np
import pytest
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import (
from xarray.tests import has_dask, requires_dask
@pytest.mark.skipif(has_dask, reason='requires dask not to be installed')
def test_dont_get_dask_if_not_installed(self) -> None:
    with pytest.raises(ValueError, match='unrecognized chunk manager dask'):
        guess_chunkmanager('dask')