import numpy as np
import pytest
import pandas as pd
from pandas import Index
@pytest.fixture(params=[pd.Timedelta('10m7s').to_pytimedelta(), pd.Timedelta('10m7s'), pd.Timedelta('10m7s').to_timedelta64()], ids=lambda x: type(x).__name__)
def scalar_td(request):
    """
    Several variants of Timedelta scalars representing 10 minutes and 7 seconds.
    """
    return request.param