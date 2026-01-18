from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
@pytest.fixture(scope='function')
def data_pd():
    cur_dir = Path(__file__).parent.resolve()
    file_path = cur_dir / 'results/mstl_elec_vic.csv'
    return pd.read_csv(file_path, index_col=['ds'], parse_dates=['ds'])