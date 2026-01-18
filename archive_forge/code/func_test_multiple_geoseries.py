import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_multiple_geoseries(self):
    """
        Additional GeoSeries need to be removed as they cannot be converted to GeoJSON
        """
    gdf = self.nybb
    gdf['boundary'] = gdf.boundary
    gdf['centroid'] = gdf.centroid
    gdf.explore()