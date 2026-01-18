import codecs
import locale
import os
import pytest
from pandas._config.localization import (
from pandas.compat import ISMUSL
import pandas as pd
@pytest.mark.parametrize('lc_var', (locale.LC_ALL, locale.LC_CTYPE, pytest.param(locale.LC_TIME, marks=pytest.mark.skipif(ISMUSL, reason='MUSL allows setting invalid LC_TIME.'))))
def test_can_set_locale_invalid_set(lc_var):
    before_locale = _get_current_locale(lc_var)
    assert not can_set_locale('non-existent_locale', lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale