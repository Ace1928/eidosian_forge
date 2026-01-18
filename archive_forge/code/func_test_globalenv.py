import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_globalenv():
    assert isinstance(rinterface.globalenv, rinterface.SexpEnvironment)