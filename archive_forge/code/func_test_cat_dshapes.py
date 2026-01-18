import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
def test_cat_dshapes():
    dslist = [dshape('3 * 10 * int32')]
    assert datashape.cat_dshapes(dslist) == dslist[0]
    dslist = [dshape('3 * 10 * int32'), dshape('7 * 10 * int32')]
    assert datashape.cat_dshapes(dslist) == dshape('10 * 10 * int32')