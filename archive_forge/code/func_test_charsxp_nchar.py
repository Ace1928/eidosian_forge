import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_charsxp_nchar():
    v = rinterface.StrSexpVector(['abc', 'de', ''])
    cs = v.get_charsxp(0)
    assert cs.nchar() == 3
    cs = v.get_charsxp(1)
    assert cs.nchar() == 2
    cs = v.get_charsxp(2)
    assert cs.nchar() == 0