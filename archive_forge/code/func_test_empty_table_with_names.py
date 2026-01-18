from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_empty_table_with_names():
    data = []
    names = ['a', 'b']
    message = 'Length of names [(]2[)] does not match length of arrays [(]0[)]'
    with pytest.raises(ValueError, match=message):
        pa.Table.from_arrays(data, names=names)