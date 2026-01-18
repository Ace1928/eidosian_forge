from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_error_on_datashape_with_string_argument():
    with pytest.raises(TypeError):
        DataShape('5 * int32')