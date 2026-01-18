import codecs
import locale
import os
import pytest
from pandas._config.localization import (
from pandas.compat import ISMUSL
import pandas as pd
def mock_get_locale():
    raise ValueError()