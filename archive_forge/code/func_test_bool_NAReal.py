import pytest
import math
import rpy2.rinterface as ri
def test_bool_NAReal():
    with pytest.raises(ValueError):
        bool(ri.NA_Real)