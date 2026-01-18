import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_login_using_explicit_credentials():
    Interface(server='http://server/', user='user', password='password')