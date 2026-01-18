import os.path as op
from uuid import uuid1
import time
from pyxnat.tests import skip_if_no_network
from pyxnat import Interface
from pyxnat.core import interfaces
@skip_if_no_network
def test_05_cleanup():
    subject.delete()
    assert not subject.exists()