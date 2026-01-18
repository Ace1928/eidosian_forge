import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_missing_arg():
    exp = rinterface.parse('function(x) { missing(x) }')
    fun = rinterface.baseenv['eval'](exp)
    nonmissing = rinterface.IntSexpVector([0])
    missing = rinterface.MissingArg
    assert not fun(nonmissing)[0]
    assert fun(missing)[0]