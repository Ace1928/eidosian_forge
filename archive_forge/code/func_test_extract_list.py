import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_extract_list():
    letters = robjects.baseenv['letters']
    myList = robjects.baseenv['list'](l=letters, f='foo')
    idem = robjects.baseenv['identical']
    assert idem(letters, myList.rx('l')[0])[0]
    assert idem('foo', myList.rx('f')[0])[0]