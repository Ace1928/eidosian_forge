from datashader import datashape
import pytest
def test_dshape_compare():
    assert datashape.int32 != datashape.dshape('1 * int32')