from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_funcproto_attrs():
    f = dshape('(int32, ?float64) -> {a: ?string}').measure
    assert f.restype == DataShape(Record([('a', Option(String()))]))
    assert f.argtypes == (DataShape(int32), DataShape(Option(float64)))