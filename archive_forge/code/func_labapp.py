import urllib.parse
import pytest
from jupyter_server.utils import url_path_join
from jupyterlab_server import LabConfig
from tornado.escape import url_escape
from traitlets import Unicode
from jupyterlab.labapp import LabApp
@pytest.fixture
def labapp(jp_serverapp, make_lab_app):
    app = make_lab_app()
    app._link_jupyter_server_extension(jp_serverapp)
    app.initialize()
    return app