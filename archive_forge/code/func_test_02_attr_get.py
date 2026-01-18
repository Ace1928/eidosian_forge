import os.path as op
from uuid import uuid1
import time
from pyxnat.tests import skip_if_no_network
from pyxnat import Interface
from pyxnat.core import interfaces
@skip_if_no_network
def test_02_attr_get():
    assert experiment.attrs.get('xnat:mrSessionData/age') == '42.0'