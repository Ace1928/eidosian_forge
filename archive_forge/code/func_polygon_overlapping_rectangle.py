import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def polygon_overlapping_rectangle():
    """Polygon overlapping rectangle"""
    wkt = 'POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))'
    geom1 = load_wkt(wkt)
    geom2 = clip_by_rect(geom1, 5, 5, 15, 15)
    assert dump_wkt(geom2, rounding_precision=0) == 'POLYGON ((5 5, 5 15, 10 15, 10 10, 15 10, 15 5, 5 5))'