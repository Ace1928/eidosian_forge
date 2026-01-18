import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_circular_repeated_point(self):
    projection = FakeProjection()
    line_string = sgeom.LineString([(0, 0), (360, 0)])
    multi_line_string = projection.project_geometry(line_string)
    assert len(multi_line_string.geoms) == 1
    assert len(multi_line_string.geoms[0].coords) == 2