from shapely.geometry import MultiPolygon, Point, Polygon
def test_multipolygon_empty_among_polygon():
    """An empty polygon passed to MultiPolygon() is ignored."""
    assert len(MultiPolygon([Point(0, 0).buffer(1.0), Polygon()]).geoms) == 1