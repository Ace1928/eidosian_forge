import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def check_array_type_preserved(self, constructor, array_type, check):
    lons, lats = np.meshgrid(np.linspace(-180, 180, 100), np.linspace(-85, 85, 100))
    lons = lons.flatten()
    lats = lats.flatten()
    array_lons = constructor(lons)
    array_lats = constructor(lats)
    self.assertIsInstance(array_lons, array_type)
    self.assertIsInstance(array_lats, array_type)
    eastings, northings = Tiles.lon_lat_to_easting_northing(array_lons, array_lats)
    self.assertIsInstance(eastings, array_type)
    self.assertIsInstance(northings, array_type)
    new_lons, new_lats = Tiles.easting_northing_to_lon_lat(eastings, northings)
    self.assertIsInstance(new_lons, array_type)
    self.assertIsInstance(new_lats, array_type)
    check(array_lons, new_lons)
    check(array_lats, new_lats)