import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, DataShapeSyntaxError
@pytest.mark.parametrize('s', ['{"./abc": int64}', '{"./a b c": float64}', '{"./a b\tc": string}', '{"./a/[0 1 2]/b/\\n": float32}'])
def test_arbitrary_string(s):
    ds = dshape(s)
    assert dshape(str(ds)) == ds