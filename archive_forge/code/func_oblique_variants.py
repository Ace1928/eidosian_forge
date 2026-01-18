from copy import deepcopy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.fixture
def oblique_variants(oblique_mercator, rotated_mercator) -> Tuple[ccrs.ObliqueMercator, ccrs.ObliqueMercator, ccrs.ObliqueMercator]:
    """Setup three ObliqueMercator objects, two identical, for eq testing."""
    default = oblique_mercator
    alt_1 = rotated_mercator
    alt_2 = deepcopy(rotated_mercator)
    return (default, alt_1, alt_2)