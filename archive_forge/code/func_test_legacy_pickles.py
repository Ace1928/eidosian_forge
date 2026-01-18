from contextlib import contextmanager
import glob
import os
import pathlib
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point
@pytest.mark.skipif(compat.USE_SHAPELY_20 or compat.USE_PYGEOS, reason='shapely 2.0/pygeos-based unpickling currently only works for shapely-2.0/pygeos-written files')
def test_legacy_pickles(current_pickle_data, legacy_pickle):
    result = pd.read_pickle(legacy_pickle)
    for name, value in result.items():
        expected = current_pickle_data[name]
        assert_geodataframe_equal(value, expected)