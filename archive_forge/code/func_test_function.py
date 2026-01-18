import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('rcode', ('function(x, y) TRUE', 'function() TRUE'))
def test_function(rcode):
    r_func = robjects.functions.Function(robjects.r(rcode))
    assert isinstance(r_func.__doc__, str)