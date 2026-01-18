import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_NULL_nonzero():
    assert not rinterface.NULL