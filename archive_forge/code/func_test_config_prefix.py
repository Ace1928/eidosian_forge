import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_config_prefix(self):
    with cf.config_prefix('base'):
        cf.register_option('a', 1, 'doc1')
        cf.register_option('b', 2, 'doc2')
        assert cf.get_option('a') == 1
        assert cf.get_option('b') == 2
        cf.set_option('a', 3)
        cf.set_option('b', 4)
        assert cf.get_option('a') == 3
        assert cf.get_option('b') == 4
    assert cf.get_option('base.a') == 3
    assert cf.get_option('base.b') == 4
    assert 'doc1' in cf.describe_option('base.a', _print_desc=False)
    assert 'doc2' in cf.describe_option('base.b', _print_desc=False)
    cf.reset_option('base.a')
    cf.reset_option('base.b')
    with cf.config_prefix('base'):
        assert cf.get_option('a') == 1
        assert cf.get_option('b') == 2