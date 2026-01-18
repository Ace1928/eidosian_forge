from datashader.datashape.predicates import isfixed, _dimensions, isnumeric, isscalar
from datashader.datashape.coretypes import TypeVar, int32, Categorical
def test_isfixed():
    assert not isfixed(TypeVar('M') * int32)