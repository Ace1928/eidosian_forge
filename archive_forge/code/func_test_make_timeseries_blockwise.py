from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
@pytest.mark.skip_with_pyarrow_strings
def test_make_timeseries_blockwise():
    df = dd.demo.make_timeseries()
    df = df[['x', 'y']]
    keys = [(df._name, i) for i in range(df.npartitions)]
    graph = optimize_dataframe_getitem(df.__dask_graph__(), keys)
    key = [k for k in graph.layers.keys() if k.startswith('make-timeseries-')][0]
    assert set(graph.layers[key].columns) == {'x', 'y'}
    graph = optimize_blockwise(df.__dask_graph__(), keys)
    layers = graph.layers
    name = list(layers.keys())[0]
    assert len(layers) == 1
    assert isinstance(layers[name], Blockwise)