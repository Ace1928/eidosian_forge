from __future__ import annotations
from typing import Literal
from pandas._libs import lib
from pandas.testing import assert_extension_array_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from pandas.testing import assert_index_equal
from pandas.testing import assert_series_equal as pd_assert_series_equal
from modin.utils import _inherit_docstrings, try_cast_to_pandas
@_inherit_docstrings(pd_assert_series_equal, apilink='pandas.testing.assert_series_equal')
def assert_series_equal(left, right, check_dtype: bool | Literal['equiv']=True, check_index_type: bool | Literal['equiv']='equiv', check_series_type: bool=True, check_names: bool=True, check_exact: bool | lib.NoDefault=lib.no_default, check_datetimelike_compat: bool=False, check_categorical: bool=True, check_category_order: bool=True, check_freq: bool=True, check_flags: bool=True, rtol: float | lib.NoDefault=lib.no_default, atol: float | lib.NoDefault=lib.no_default, obj: str='Series', *, check_index: bool=True, check_like: bool=False) -> None:
    left = try_cast_to_pandas(left)
    right = try_cast_to_pandas(right)
    pd_assert_series_equal(left, right, check_dtype=check_dtype, check_index_type=check_index_type, check_series_type=check_series_type, check_names=check_names, check_exact=check_exact, check_datetimelike_compat=check_datetimelike_compat, check_categorical=check_categorical, check_category_order=check_category_order, check_freq=check_freq, check_flags=check_flags, rtol=rtol, atol=atol, obj=obj, check_index=check_index, check_like=check_like)