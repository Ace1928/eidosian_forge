from __future__ import annotations
import pytest
import dask
def test_full_dataset():
    pytest.importorskip('mimesis')
    b = dask.datasets.make_people(npartitions=2, records_per_partition=10)
    assert b.count().compute() == 20