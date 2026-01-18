from datashader import datashape
import pytest
def test_custom_typeset_repr():
    mytypeset = datashape.TypeSet(datashape.int64, datashape.float64)
    assert repr(mytypeset).startswith('TypeSet(')
    assert repr(mytypeset).endswith('name=None)')