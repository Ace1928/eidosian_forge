from uuid import uuid1
from pyxnat import Interface
from pyxnat import jsonutil
import os.path as op
def test_datatypes():
    assert 'xnat:subjectData' in central.inspect.datatypes()