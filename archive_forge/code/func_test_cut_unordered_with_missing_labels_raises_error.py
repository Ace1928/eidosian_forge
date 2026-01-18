import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_unordered_with_missing_labels_raises_error():
    msg = "'labels' must be provided if 'ordered = False'"
    with pytest.raises(ValueError, match=msg):
        cut([0.5, 3], bins=[0, 1, 2], ordered=False)