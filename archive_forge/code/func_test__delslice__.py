import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test__delslice__(self):
    tl = rlc.TaggedList((1, 2, 3, 4), tags=('a', 'b', 'c', 'd'))
    del tl[1:3]
    assert len(tl) == 2
    assert tl.tags == ('a', 'd')
    assert tuple(tl) == (1, 4)