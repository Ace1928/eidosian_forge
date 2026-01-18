from unittest import SkipTest
import numpy as np
import pandas as pd
from shapely import geometry as sgeom
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path, Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import GeomTests
from geoviews.data import GeoPandasInterface
from .test_multigeometry import GeomInterfaceTest
def test_geometry_column_not_named_geometry(self):
    gdf = geopandas.GeoDataFrame({'v': [1, 2], 'not geometry': [sgeom.Point(0, 1), sgeom.Point(1, 2)]}, geometry='not geometry')
    ds = Dataset(gdf, kdims=['Longitude', 'Latitude'], datatype=[self.datatype])
    self.assertEqual(ds.dimension_values('Longitude'), np.array([0, 1]))
    self.assertEqual(ds.dimension_values('Latitude'), np.array([1, 2]))