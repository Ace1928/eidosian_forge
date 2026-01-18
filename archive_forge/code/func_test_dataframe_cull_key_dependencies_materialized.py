from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def test_dataframe_cull_key_dependencies_materialized():
    datasets = pytest.importorskip('dask.datasets')
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.skip('not supported')
    ddf = datasets.timeseries(end='2000-01-15')
    name = 'custom_graph_test'
    name_0 = 'custom_graph_test_0'
    dsk = {}
    for i in range(ddf.npartitions):
        dsk[name_0, i] = (lambda x: x, (ddf._name, i))
        dsk[name, i] = (lambda x: x, (name_0, i))
    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
    result = dd.core.new_dd_object(dsk, name, ddf._meta, ddf.divisions)
    graph = result.dask
    culled_keys = [k for k in result.__dask_keys__() if k != (name, 0)]
    culled_graph = graph.cull(culled_keys)
    cached_deps = culled_graph.key_dependencies.copy()
    deps = culled_graph.get_all_dependencies()
    assert cached_deps == deps
    deps0 = graph.get_all_dependencies()
    for name, i in list(deps0.keys()):
        if i == 0:
            deps0.pop((name, i))
    assert deps0 == deps