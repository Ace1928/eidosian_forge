import os.path as op
import tempfile
from pyxnat import jsonutil
def test_get_item_int_value():
    assert jtable.data[0] == jtable[0].data[0]