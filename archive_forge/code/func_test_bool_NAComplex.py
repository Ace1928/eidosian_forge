import pytest
import math
import rpy2.rinterface as ri
def test_bool_NAComplex():
    with pytest.raises(ValueError):
        bool(ri.NA_Complex)