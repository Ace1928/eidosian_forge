import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def polygon_fully_within_rectangle():
    """Polygon fully within rectangle"""
    wkt = 'POLYGON ((1 1, 1 30, 30 30, 30 1, 1 1), (10 10, 20 10, 20 20, 10 20, 10 10))'
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 0, 0, 40, 40)
    assert dump_wkt(geom2, rounding_precision=0) == wkt