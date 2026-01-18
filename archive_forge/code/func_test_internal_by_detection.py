import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('columns', [[(True, 'a'), (True, 'b'), (True, 'c')], [(True, 'a'), (True, 'b')], [(False, 'a'), (False, 'b'), (True, 'c')], [(False, 'a'), (True, 'c')], [(False, 'a'), (True, 'c'), (False, [1, 1, 2])], [(False, 'a'), (False, 'b'), (False, 'c')], [(False, 'a'), (False, 'b'), (False, 'c'), (False, [1, 1, 2])]])
def test_internal_by_detection(columns):
    data = {'a': [1, 1, 2], 'b': [11, 11, 22], 'c': [111, 111, 222]}
    md_df = pd.DataFrame(data)
    _, by = get_external_groupers(md_df, columns, add_plus_one=True)
    md_grp = md_df.groupby(by)
    ref = frozenset((col for is_lookup, col in columns if not is_lookup and hashable(col)))
    exp = frozenset(md_grp._internal_by)
    assert ref == exp