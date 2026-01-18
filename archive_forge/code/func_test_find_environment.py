import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_environment():
    ge_R = rinterface.globalenv.find('.GlobalEnv')
    assert isinstance(ge_R, rinterface.SexpEnvironment)