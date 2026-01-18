import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_error_in_call():
    r_sum = rinterface.baseenv['sum']
    with pytest.raises(rinterface.embedded.RRuntimeError), pytest.warns(rinterface.RRuntimeWarning):
        r_sum(2, 'a')