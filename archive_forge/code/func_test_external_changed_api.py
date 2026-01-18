from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@dec.skip_without('pandas')
def test_external_changed_api(monkeypatch):
    """Check that the execution rejects if external API changed paths"""
    import pandas as pd
    series = pd.Series([1], index=['a'])
    with monkeypatch.context() as m:
        m.delattr(pd, 'Series')
        context = limited(data=series)
        with pytest.raises(GuardRejection):
            guarded_eval('data.iloc[0]', context)