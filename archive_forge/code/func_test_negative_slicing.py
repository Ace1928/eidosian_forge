from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_negative_slicing():
    ds = dshape('10 * int')
    assert ds.subshape[-3:] == dshape('3 * int')