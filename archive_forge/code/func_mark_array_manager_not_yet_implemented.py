import pandas.util._test_decorators as td
from __future__ import annotations
import locale
from typing import (
import pytest
from pandas._config import get_option
from pandas._config.config import _get_option
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
def mark_array_manager_not_yet_implemented(request) -> None:
    mark = pytest.mark.xfail(reason='Not yet implemented for ArrayManager')
    request.applymarker(mark)