import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def test_check_lonlat_to_eastingnorthing_identity(self):
    for lon in np.linspace(-180, 180, 100):
        for lat in np.linspace(-85, 85, 100):
            easting, northing = Tiles.lon_lat_to_easting_northing(lon, lat)
            new_lon, new_lat = Tiles.easting_northing_to_lon_lat(easting, northing)
            self.assertAlmostEqual(lon, new_lon, places=2)
            self.assertAlmostEqual(lat, new_lat, places=2)