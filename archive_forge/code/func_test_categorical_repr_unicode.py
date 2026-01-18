from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_repr_unicode(self):

    class County:
        name = 'San Sebastián'
        state = 'PR'

        def __repr__(self) -> str:
            return self.name + ', ' + self.state
    cat = Categorical([County() for _ in range(61)])
    idx = Index(cat)
    ser = idx.to_series()
    repr(ser)
    str(ser)