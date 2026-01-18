from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@dec.skip_without('pandas')
def test_accepts_non_overriden_properties():
    import pandas as pd

    class GoodProperty(pd.Series):
        pass
    series = GoodProperty([1], index=['a'])
    context = limited(data=series)
    assert guarded_eval('data.iloc[0]', context) == 1