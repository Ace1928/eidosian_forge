import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_get_option(self):
    cf.register_option('a', 1, 'doc')
    cf.register_option('b.c', 'hullo', 'doc2')
    cf.register_option('b.b', None, 'doc2')
    assert cf.get_option('a') == 1
    assert cf.get_option('b.c') == 'hullo'
    assert cf.get_option('b.b') is None
    msg = "No such keys\\(s\\): 'no_such_option'"
    with pytest.raises(OptionError, match=msg):
        cf.get_option('no_such_option')