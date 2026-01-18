import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test__init__(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    with pytest.raises(ValueError):
        rlc.TaggedList((1, 2, 3), tags=('b', 'c'))