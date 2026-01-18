import numpy as np
import pandas
import pytest
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
@pytest.mark.parametrize('pandas_dtype, c_string', [(np.dtype('bool'), 'b'), (np.dtype('int8'), 'c'), (np.dtype('uint8'), 'C'), (np.dtype('int16'), 's'), (np.dtype('uint16'), 'S'), (np.dtype('int32'), 'i'), (np.dtype('uint32'), 'I'), (np.dtype('int64'), 'l'), (np.dtype('uint64'), 'L'), (np.dtype('float16'), 'e'), (np.dtype('float32'), 'f'), (np.dtype('float64'), 'g'), (pandas.Series(['a']).dtype, 'u'), (pandas.Series([0]).astype('datetime64[ns]').dtype, 'tsn:')])
def test_dtype_to_arrow_c(pandas_dtype, c_string):
    """Test ``pandas_dtype_to_arrow_c`` utility function."""
    assert pandas_dtype_to_arrow_c(pandas_dtype) == c_string