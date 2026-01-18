import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test__setslice__(self):
    tl = rlc.TaggedList((1, 2, 3, 4), tags=('a', 'b', 'c', 'd'))
    tl[1:3] = [5, 6]
    assert len(tl) == 4
    assert tl.tags == ('a', 'b', 'c', 'd')
    assert tuple(tl) == (1, 5, 6, 4)