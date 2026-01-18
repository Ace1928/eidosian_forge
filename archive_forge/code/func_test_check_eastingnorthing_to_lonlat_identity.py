import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def test_check_eastingnorthing_to_lonlat_identity(self):
    for easting in np.linspace(-20037508.34, 20037508.34, 100):
        for northing in np.linspace(-20037508.34, 20037508.34, 100):
            lon, lat = Tiles.easting_northing_to_lon_lat(easting, northing)
            new_easting, new_northing = Tiles.lon_lat_to_easting_northing(lon, lat)
            self.assertAlmostEqual(easting, new_easting, places=2)
            self.assertAlmostEqual(northing, new_northing, places=2)