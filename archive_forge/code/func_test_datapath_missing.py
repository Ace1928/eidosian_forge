import os
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('strict_data_files', [True, False])
def test_datapath_missing(datapath):
    with pytest.raises(ValueError, match='Could not find file'):
        datapath('not_a_file')