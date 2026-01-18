import codecs
import locale
import os
import pytest
from pandas._config.localization import (
from pandas.compat import ISMUSL
import pandas as pd
def test_can_set_locale_invalid_get(monkeypatch):

    def mock_get_locale():
        raise ValueError()
    with monkeypatch.context() as m:
        m.setattr(locale, 'getlocale', mock_get_locale)
        assert not can_set_locale('')