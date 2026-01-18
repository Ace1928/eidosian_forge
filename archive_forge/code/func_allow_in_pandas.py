import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_object_dtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.tests.extension import base
@pytest.fixture
def allow_in_pandas(monkeypatch):
    """
    A monkeypatch to tells pandas to let us in.

    By default, passing a NumpyExtensionArray to an index / series / frame
    constructor will unbox that NumpyExtensionArray to an ndarray, and treat
    it as a non-EA column. We don't want people using EAs without
    reason.

    The mechanism for this is a check against ABCNumpyExtensionArray
    in each constructor.

    But, for testing, we need to allow them in pandas. So we patch
    the _typ of NumpyExtensionArray, so that we evade the ABCNumpyExtensionArray
    check.
    """
    with monkeypatch.context() as m:
        m.setattr(NumpyExtensionArray, '_typ', 'extension')
        m.setattr(tm.asserters, 'assert_attr_equal', _assert_attr_equal)
        yield