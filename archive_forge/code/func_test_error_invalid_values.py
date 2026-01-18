import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_error_invalid_values(data, all_arithmetic_operators, using_infer_string):
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)
    if using_infer_string:
        import pyarrow as pa
        errs = (TypeError, pa.lib.ArrowNotImplementedError, NotImplementedError)
    else:
        errs = TypeError
    msg = '|'.join(['can only perform ops with numeric values', 'IntegerArray cannot perform the operation mod', 'unsupported operand type', 'can only concatenate str \\(not \\"int\\"\\) to str', 'not all arguments converted during string', "ufunc '.*' not supported for the input types, and the inputs could not", "ufunc '.*' did not contain a loop with signature matching types", 'Addition/subtraction of integers and integer-arrays with Timestamp', 'has no kernel', 'not implemented'])
    with pytest.raises(errs, match=msg):
        ops('foo')
    with pytest.raises(errs, match=msg):
        ops(pd.Timestamp('20180101'))
    str_ser = pd.Series('foo', index=s.index)
    if all_arithmetic_operators in ['__mul__', '__rmul__'] and (not using_infer_string):
        res = ops(str_ser)
        expected = pd.Series(['foo' * x for x in data], index=s.index)
        expected = expected.fillna(np.nan)
        tm.assert_series_equal(res, expected)
    else:
        with pytest.raises(errs, match=msg):
            ops(str_ser)
    msg = '|'.join(['can only perform ops with numeric values', 'cannot perform .* with this index type: DatetimeArray', 'Addition/subtraction of integers and integer-arrays with DatetimeArray is no longer supported. *', 'unsupported operand type', 'can only concatenate str \\(not \\"int\\"\\) to str', 'not all arguments converted during string', 'cannot subtract DatetimeArray from ndarray', 'has no kernel', 'not implemented'])
    with pytest.raises(errs, match=msg):
        ops(pd.Series(pd.date_range('20180101', periods=len(s))))