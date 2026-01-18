import os.path as op
import tempfile
from pyxnat import jsonutil
def test_get_item_list_value():
    headers = set(jtable[['projects', 'subjectid']].headers())
    assert headers == set(['projects', 'subjectid'])