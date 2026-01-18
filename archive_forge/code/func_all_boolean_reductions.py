import operator
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_series_equal
from pandas.tests.extension import base as extension_tests
import shapely.geometry
from shapely.geometry import Point
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import (
import pytest
@pytest.fixture(params=['all', 'any'])
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names
    """
    return request.param