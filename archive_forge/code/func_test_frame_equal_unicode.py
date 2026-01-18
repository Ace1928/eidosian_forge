import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('df1,df2,msg', [(DataFrame({'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'ë']}), DataFrame({'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'e̊']}), '{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) are different\n\n{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) values are different \\(33\\.33333 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[é, è, ë\\]\n\\[right\\]: \\[é, è, e̊\\]'), (DataFrame({'A': ['á', 'à', 'ä'], 'E': ['é', 'è', 'ë']}), DataFrame({'A': ['a', 'a', 'a'], 'E': ['e', 'e', 'e']}), '{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) are different\n\n{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) values are different \\(100\\.0 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\[á, à, ä\\]\n\\[right\\]: \\[a, a, a\\]')])
def test_frame_equal_unicode(df1, df2, msg, by_blocks_fixture, obj_fixture):
    msg = msg.format(obj=obj_fixture)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj_fixture)