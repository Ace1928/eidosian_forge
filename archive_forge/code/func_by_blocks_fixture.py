import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture(params=[True, False])
def by_blocks_fixture(request):
    return request.param