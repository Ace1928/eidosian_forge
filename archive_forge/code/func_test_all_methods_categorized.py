from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
def test_all_methods_categorized(mframe):
    grp = mframe.groupby(mframe.iloc[:, 0])
    names = {_ for _ in dir(grp) if not _.startswith('_')} - set(mframe.columns)
    new_names = set(names)
    new_names -= reduction_kernels
    new_names -= transformation_kernels
    new_names -= groupby_other_methods
    assert not reduction_kernels & transformation_kernels
    assert not reduction_kernels & groupby_other_methods
    assert not transformation_kernels & groupby_other_methods
    if new_names:
        msg = f'\nThere are uncategorized methods defined on the Grouper class:\n{new_names}.\n\nWas a new method recently added?\n\nEvery public method On Grouper must appear in exactly one the\nfollowing three lists defined in pandas.core.groupby.base:\n- `reduction_kernels`\n- `transformation_kernels`\n- `groupby_other_methods`\nsee the comments in pandas/core/groupby/base.py for guidance on\nhow to fix this test.\n        '
        raise AssertionError(msg)
    all_categorized = reduction_kernels | transformation_kernels | groupby_other_methods
    if names != all_categorized:
        msg = f"\nSome methods which are supposed to be on the Grouper class\nare missing:\n{all_categorized - names}.\n\nThey're still defined in one of the lists that live in pandas/core/groupby/base.py.\nIf you removed a method, you should update them\n"
        raise AssertionError(msg)