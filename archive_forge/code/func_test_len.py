import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_len(self):
    x = rlc.OrdDict()
    assert len(x) == 0
    x['a'] = 2
    x['b'] = 1
    assert len(x) == 2