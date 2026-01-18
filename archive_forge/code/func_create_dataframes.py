import itertools
import numpy as np
import pytest
from pandas import (
def create_dataframes():
    return [DataFrame(columns=['a', 'a']), DataFrame(np.arange(15).reshape((5, 3)), columns=['a', 'a', 99])] + [DataFrame(s) for s in create_series()]