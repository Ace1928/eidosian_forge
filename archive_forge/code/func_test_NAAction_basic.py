import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def test_NAAction_basic():
    import pytest
    pytest.raises(ValueError, NAAction, on_NA='pord')
    pytest.raises(ValueError, NAAction, NA_types=('NaN', 'asdf'))
    pytest.raises(ValueError, NAAction, NA_types='NaN')
    assert_no_pickling(NAAction())