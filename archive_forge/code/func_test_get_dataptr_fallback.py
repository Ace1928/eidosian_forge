import pytest
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface
def test_get_dataptr_fallback():
    with pytest.raises(NotImplementedError):
        openrlib._get_dataptr_fallback(None)