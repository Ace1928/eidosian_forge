import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_close_jsession():
    with Interface(config=fp) as central_intf:
        assert central_intf.select.project('OASIS3').exists()