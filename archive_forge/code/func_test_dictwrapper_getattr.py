import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
def test_dictwrapper_getattr(self):
    options = cf.options
    with pytest.raises(OptionError, match='No such option'):
        options.bananas
    assert not hasattr(options, 'bananas')