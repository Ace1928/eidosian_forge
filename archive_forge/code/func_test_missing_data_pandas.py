import numpy as np
import pandas
from statsmodels.tools import data
def test_missing_data_pandas():
    """
    Fixes GH: #144
    """
    X = np.random.random((10, 5))
    X[1, 2] = np.nan
    df = pandas.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    np.testing.assert_equal(rnames.tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9])