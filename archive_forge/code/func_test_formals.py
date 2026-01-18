import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('rcode,funcnames', (('function(x, y) TRUE', ('x', 'y')), ('.C', None), ('`if`', None)))
def test_formals(rcode, funcnames):
    ri_f = robjects.r(rcode)
    res = ri_f.formals()
    if funcnames:
        res = robjects.r['as.list'](res)
        assert len(res) == len(funcnames)
        assert funcnames == tuple(res.names)
    else:
        res is None