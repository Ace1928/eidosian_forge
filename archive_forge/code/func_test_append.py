import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_append(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    assert len(tl) == 3
    tl.append(4, tag='a')
    assert len(tl) == 4
    assert tl[3] == 4
    assert tl.tags == ('a', 'b', 'c', 'a')