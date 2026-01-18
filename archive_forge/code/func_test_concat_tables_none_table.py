from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_concat_tables_none_table():
    with pytest.raises(AttributeError):
        pa.concat_tables([None])