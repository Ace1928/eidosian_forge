import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture()
def column_group_df():
    return pd.DataFrame([[0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 1, 0, 2]], columns=['A', 'B', 'C', 'D', 'E', 'F', 'G'])