import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_vector():
    pi_R = rinterface.globalenv.find('pi')
    assert isinstance(pi_R, rinterface.SexpVector)