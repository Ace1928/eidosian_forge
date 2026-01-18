import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_frame():
    env = rinterface.baseenv['new.env']()
    f = env.frame()
    assert f is rinterface.NULL