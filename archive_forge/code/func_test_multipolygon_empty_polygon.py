from shapely.geometry import MultiPolygon, Point, Polygon
def test_multipolygon_empty_polygon():
    """An empty polygon passed to MultiPolygon() makes an empty
    multipolygon geometry."""
    assert MultiPolygon([Polygon()]).is_empty