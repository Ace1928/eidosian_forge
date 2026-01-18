import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_metadata():
    df = MySubclassWithMetadata(np.random.default_rng(2).random((5, 3)), columns=['A', 'B', 'C'])
    subset = df[['A', 'B']]
    assert isinstance(subset, MySubclassWithMetadata)