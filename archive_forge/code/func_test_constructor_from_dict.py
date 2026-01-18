import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_constructor_from_dict():
    result = SubclassedSeries({'a': 1, 'b': 2, 'c': 3})
    assert isinstance(result, SubclassedSeries)