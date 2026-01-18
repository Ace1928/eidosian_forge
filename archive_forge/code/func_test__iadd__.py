import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test__iadd__(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    tl += tl
    assert len(tl) == 6
    assert tl.tags == ('a', 'b', 'c', 'a', 'b', 'c')
    assert tuple(tl) == (1, 2, 3, 1, 2, 3)