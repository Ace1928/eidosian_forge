import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_iterontag(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'a'))
    assert tuple(tl.iterontag('a')) == (1, 3)