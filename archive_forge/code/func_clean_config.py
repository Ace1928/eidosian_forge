import pytest
from pandas._config import config as cf
from pandas._config.config import OptionError
import pandas as pd
import pandas._testing as tm
@pytest.fixture(autouse=True)
def clean_config(self, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(cf, '_global_config', {})
        m.setattr(cf, 'options', cf.DictWrapper(cf._global_config))
        m.setattr(cf, '_deprecated_options', {})
        m.setattr(cf, '_registered_options', {})
        cf.register_option('chained_assignment', 'raise')
        yield