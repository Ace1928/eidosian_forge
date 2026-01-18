import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_instantiate_crs_str():
    ccrs = pytest.importorskip('cartopy.crs')
    assert isinstance(instantiate_crs_str('PlateCarree'), ccrs.PlateCarree)