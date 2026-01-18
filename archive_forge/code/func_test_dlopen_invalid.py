import pytest
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface
def test_dlopen_invalid():
    with pytest.raises(ValueError):
        openrlib._dlopen_rlib(None)