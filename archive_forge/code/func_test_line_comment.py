from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('read_kwargs', [{}, {'lineterminator': '*'}, {'delim_whitespace': True}])
def test_line_comment(all_parsers, read_kwargs, request):
    parser = all_parsers
    data = '# empty\nA,B,C\n1,2.,4.#hello world\n#ignore this line\n5.,NaN,10.0\n'
    warn = None
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    if read_kwargs.get('delim_whitespace'):
        data = data.replace(',', ' ')
        warn = FutureWarning
    elif read_kwargs.get('lineterminator'):
        data = data.replace('\n', read_kwargs.get('lineterminator'))
    read_kwargs['comment'] = '#'
    if parser.engine == 'pyarrow':
        if 'lineterminator' in read_kwargs:
            msg = "The 'lineterminator' option is not supported with the 'pyarrow' engine"
        else:
            msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), **read_kwargs)
        return
    elif parser.engine == 'python' and read_kwargs.get('lineterminator'):
        msg = 'Custom line terminators not supported in python parser \\(yet\\)'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), **read_kwargs)
        return
    with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), **read_kwargs)
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)