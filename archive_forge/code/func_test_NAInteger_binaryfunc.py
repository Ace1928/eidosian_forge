import pytest
import math
import rpy2.rinterface as ri
@pytest.mark.skip(reason='Python changed the behavior for int-inheriting objects.')
def test_NAInteger_binaryfunc():
    na_int = ri.NAInteger
    assert na_int + 2 is na_int