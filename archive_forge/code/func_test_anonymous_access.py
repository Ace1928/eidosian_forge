import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_anonymous_access():
    projects = central_anon.select.projects().get()
    assert isinstance(projects, list)
    assert list