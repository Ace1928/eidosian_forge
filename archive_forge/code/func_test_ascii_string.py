from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_ascii_string(self):
    assert from_numpy((2,), np.dtype('S7')) == dshape('2 * string[7, "ascii"]')