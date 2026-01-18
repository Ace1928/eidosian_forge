import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_missingtype():
    assert not rinterface.MissingArg