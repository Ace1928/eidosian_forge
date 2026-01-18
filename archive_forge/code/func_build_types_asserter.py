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
def build_types_asserter(comparator):

    def wrapper(obj1, obj2, *args, **kwargs):
        error_str = f'obj1 and obj2 has incorrect types: {type(obj1)} and {type(obj2)}'
        assert not is_scalar(obj1) ^ is_scalar(obj2), error_str
        assert obj1.__module__.split('.')[0] == 'modin', error_str
        assert obj2.__module__.split('.')[0] == 'pandas', error_str
        comparator(obj1, obj2, *args, **kwargs)
    return wrapper