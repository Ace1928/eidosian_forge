from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('comment_char', ['#', '~', '&', '^', '*', '@'])
def test_custom_comment_char(all_parsers, comment_char):
    parser = all_parsers
    data = 'a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo'
    if parser.engine == 'pyarrow':
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data.replace('#', comment_char)), comment=comment_char)
        return
    result = parser.read_csv(StringIO(data.replace('#', comment_char)), comment=comment_char)
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)