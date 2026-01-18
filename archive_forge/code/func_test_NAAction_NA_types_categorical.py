import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def test_NAAction_NA_types_categorical():
    for NA_types in [[], ['NaN'], ['None'], ['NaN', 'None']]:
        action = NAAction(NA_types=NA_types)
        assert not action.is_categorical_NA('a')
        assert not action.is_categorical_NA(1)
        assert action.is_categorical_NA(None) == ('None' in NA_types)
        assert action.is_categorical_NA(np.nan) == ('NaN' in NA_types)