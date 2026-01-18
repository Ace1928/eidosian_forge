import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_enclos_get():
    assert isinstance(rinterface.baseenv.enclos, rinterface.SexpEnvironment)
    env = rinterface.baseenv['new.env']()
    assert isinstance(env.enclos, rinterface.SexpEnvironment)