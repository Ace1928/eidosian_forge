import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_instantiate_crs_str_kwargs():
    ccrs = pytest.importorskip('cartopy.crs')
    crs = instantiate_crs_str('PlateCarree', globe=ccrs.Globe(datum='WGS84'))
    assert isinstance(crs, ccrs.PlateCarree)
    assert isinstance(crs.globe, ccrs.Globe)
    assert crs.globe.datum == 'WGS84'