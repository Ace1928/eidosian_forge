import os.path as op
import tempfile
from pyxnat import jsonutil
def test_get_item_str_value():
    jtable.items()
    assert jtable['subject_label'] == jtable.get('subject_label')