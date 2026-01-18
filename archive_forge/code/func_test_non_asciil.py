import pytest
import rpy2.rinterface as ri
def test_non_asciil():
    u_char = '↧'
    b_char = b'\xe2\x86\xa7'
    assert b_char == u_char.encode('utf-8')
    sexp = ri.StrSexpVector((u'↧',))
    char = sexp[0]
    assert isinstance(char, str)
    assert u'↧'.encode('utf-8') == char.encode('utf-8')
    sexp = ri.evalr('c("ěščřžýáíé", "abc", "百折不撓")')
    assert all((isinstance(x, str) for x in sexp))