import pytest
import math
import rpy2.rinterface as ri
def test_bool_NAInteger():
    with pytest.raises(ValueError):
        bool(ri.NA_Integer)