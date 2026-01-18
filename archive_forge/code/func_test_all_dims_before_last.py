from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_all_dims_before_last():
    with pytest.raises(TypeError):
        DataShape(uint32, var, uint32)