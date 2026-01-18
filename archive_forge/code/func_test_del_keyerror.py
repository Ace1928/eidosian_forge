import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_del_keyerror():
    with pytest.raises(KeyError):
        rinterface.globalenv.__delitem__('foo')