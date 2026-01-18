from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_values', [['-999.0', '-999'], [-999, -999.0], [-999.0, -999], ['-999.0'], ['-999'], [-999.0], [-999]])
@pytest.mark.parametrize('data', ['A,B\n-999,1.2\n2,-999\n3,4.5\n', 'A,B\n-999,1.200\n2,-999.000\n3,4.500\n'])
def test_non_string_na_values(all_parsers, data, na_values, request):
    parser = all_parsers
    expected = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=['A', 'B'])
    if parser.engine == 'pyarrow' and (not all((isinstance(x, str) for x in na_values))):
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    elif parser.engine == 'pyarrow' and '-999.000' in data:
        mark = pytest.mark.xfail(reason='pyarrow engined does not recognize equivalent floats')
        request.applymarker(mark)
    result = parser.read_csv(StringIO(data), na_values=na_values)
    tm.assert_frame_equal(result, expected)