import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture(params=['DataFrame', 'Series'])
def obj_fixture(request):
    return request.param