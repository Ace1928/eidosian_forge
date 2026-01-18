from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('compare', _get_overlap_public_nat_methods(Timestamp, True) + _get_overlap_public_nat_methods(Timedelta, True), ids=lambda x: f'{x[0].__name__}.{x[1]}')
def test_nat_doc_strings(compare):
    klass, method = compare
    klass_doc = getattr(klass, method).__doc__
    if klass == Timestamp and method == 'isoformat':
        pytest.skip("Ignore differences with Timestamp.isoformat() as they're intentional")
    if method == 'to_numpy':
        pytest.skip(f'different docstring for {method} is intentional')
    nat_doc = getattr(NaT, method).__doc__
    assert klass_doc == nat_doc