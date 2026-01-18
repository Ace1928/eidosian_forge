import pytest
from datashader.datashape.user import issubschema, validate
from datashader.datashape import dshape
from datetime import date, time, datetime
import numpy as np
@min_np
def test_nested_iteratables():
    assert validate('2 * 3 * int', [(1, 2, 3), (4, 5, 6)])