import codecs
import locale
import os
import pytest
from pandas._config.localization import (
from pandas.compat import ISMUSL
import pandas as pd
def test_encoding_detected():
    system_locale = os.environ.get('LC_ALL')
    system_encoding = system_locale.split('.')[-1] if system_locale else 'utf-8'
    assert codecs.lookup(pd.options.display.encoding).name == codecs.lookup(system_encoding).name