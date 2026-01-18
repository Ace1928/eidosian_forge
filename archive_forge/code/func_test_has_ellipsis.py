import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
@pytest.mark.parametrize('ds', [dshape('... * float32'), dshape('A... * float32'), dshape('var * ... * float32'), dshape('(int32, M... * int16) -> var * int8'), dshape('(int32, var * int16) -> ... * int8'), dshape('10 * { f0: int32, f1: A... * float32 }'), dshape('{ f0 : { g0 : ... * int }, f1: int32 }'), (dshape('... * int32'),)])
def test_has_ellipsis(ds):
    assert has_ellipsis(ds)