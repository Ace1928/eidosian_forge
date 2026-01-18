import array
import pytest
import rpy2.robjects as robjects
def test_do_slot():
    assert robjects.globalenv.find('BOD').do_slot('reference')[0] == 'A1.4, p. 270'