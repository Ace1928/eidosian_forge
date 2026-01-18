import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_case_insensitive(self):
    cf.register_option('KanBAN', 1, 'doc')
    assert 'doc' in cf.describe_option('kanbaN', _print_desc=False)
    assert cf.get_option('kanBaN') == 1
    cf.set_option('KanBan', 2)
    assert cf.get_option('kAnBaN') == 2
    msg = "No such keys\\(s\\): 'no_such_option'"
    with pytest.raises(OptionError, match=msg):
        cf.get_option('no_such_option')
    cf.deprecate_option('KanBan')
    assert cf._is_deprecated('kAnBaN')