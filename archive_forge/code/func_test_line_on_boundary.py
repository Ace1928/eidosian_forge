import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def test_line_on_boundary():
    """Line on boundary"""
    geom1 = load_wkt('LINESTRING (10 15, 10 10, 15 10)')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'GEOMETRYCOLLECTION EMPTY'