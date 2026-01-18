import pytest
from rpy2 import rinterface
from rpy2.rinterface import embedded
from rpy2.rinterface_lib import sexp
@pytest.mark.skipif(embedded.rpy2_embeddedR_isinitialized, reason='Can only be tested before R is initialized.')
def test_assert_isready():
    with pytest.raises(embedded.RNotReadyError):
        embedded.assert_isready()