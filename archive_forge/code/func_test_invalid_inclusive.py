import numpy as np
import pytest
from pandas.util._validators import validate_inclusive
import pandas as pd
@pytest.mark.parametrize('invalid_inclusive', ('ccc', 2, object(), None, np.nan, pd.NA, pd.DataFrame()))
def test_invalid_inclusive(invalid_inclusive):
    with pytest.raises(ValueError, match="Inclusive has to be either 'both', 'neither', 'left' or 'right'"):
        validate_inclusive(invalid_inclusive)