from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
def create_sel_results(x_indexer, x_index, other_vars, drop_coords, drop_indexes, rename_dims):
    dim_indexers = {'x': x_indexer}
    index_vars = x_index.create_variables()
    indexes = {k: x_index for k in index_vars}
    variables = {}
    variables.update(index_vars)
    variables.update(other_vars)
    return indexing.IndexSelResult(dim_indexers=dim_indexers, indexes=indexes, variables=variables, drop_coords=drop_coords, drop_indexes=drop_indexes, rename_dims=rename_dims)