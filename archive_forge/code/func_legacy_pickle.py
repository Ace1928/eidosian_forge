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
@pytest.fixture(params=files, ids=[p.split('/')[-1] for p in files])
def legacy_pickle(request):
    return request.param