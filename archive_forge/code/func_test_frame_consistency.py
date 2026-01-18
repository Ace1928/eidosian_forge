import inspect
import pytest
from pandas import (
from pandas.core.groupby.generic import (
def test_frame_consistency(request, groupby_func):
    if groupby_func in ('first', 'last'):
        msg = 'first and last are entirely different between frame and groupby'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    if groupby_func in ('cumcount',):
        msg = 'DataFrame has no such method'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    if groupby_func == 'ngroup':
        assert not hasattr(DataFrame, groupby_func)
        return
    frame_method = getattr(DataFrame, groupby_func)
    gb_method = getattr(DataFrameGroupBy, groupby_func)
    result = set(inspect.signature(gb_method).parameters)
    if groupby_func == 'size':
        expected = {'self'}
    else:
        expected = set(inspect.signature(frame_method).parameters)
    exclude_expected, exclude_result = (set(), set())
    if groupby_func in ('any', 'all'):
        exclude_expected = {'kwargs', 'bool_only', 'axis'}
    elif groupby_func in ('count',):
        exclude_expected = {'numeric_only', 'axis'}
    elif groupby_func in ('nunique',):
        exclude_expected = {'axis'}
    elif groupby_func in ('max', 'min'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
        exclude_result = {'min_count', 'engine', 'engine_kwargs'}
    elif groupby_func in ('mean', 'std', 'sum', 'var'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
        exclude_result = {'engine', 'engine_kwargs'}
    elif groupby_func in ('median', 'prod', 'sem'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
    elif groupby_func in ('backfill', 'bfill', 'ffill', 'pad'):
        exclude_expected = {'downcast', 'inplace', 'axis'}
    elif groupby_func in ('cummax', 'cummin'):
        exclude_expected = {'skipna', 'args'}
        exclude_result = {'numeric_only'}
    elif groupby_func in ('cumprod', 'cumsum'):
        exclude_expected = {'skipna'}
    elif groupby_func in ('pct_change',):
        exclude_expected = {'kwargs'}
        exclude_result = {'axis'}
    elif groupby_func in ('rank',):
        exclude_expected = {'numeric_only'}
    elif groupby_func in ('quantile',):
        exclude_expected = {'method', 'axis'}
    assert result & exclude_result == exclude_result
    assert expected & exclude_expected == exclude_expected
    result -= exclude_result
    expected -= exclude_expected
    assert result == expected