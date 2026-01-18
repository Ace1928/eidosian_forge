import os
import os.path as op
import tempfile 
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
def test_connection_with_explicit_parameters():
    import json
    cfg = json.load(open(fp))
    Interface(server=cfg['server'], user=cfg['user'], password=cfg['password'])