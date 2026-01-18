import operator
import pytest
from pandas._config.config import _get_option
from pandas import (
@pytest.fixture(params=['ffill', 'bfill'])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param