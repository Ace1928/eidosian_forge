from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_delete_search():
    central.manage.search.delete(search_name)
    assert search_name not in central.manage.search.saved()