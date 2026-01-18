from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_fixed_dimensions_must_be_positive():
    with pytest.raises(ValueError):
        Fixed(-1)