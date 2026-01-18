import urllib.parse
import pytest
from jupyter_server.utils import url_path_join
from jupyterlab_server import LabConfig
from tornado.escape import url_escape
from traitlets import Unicode
from jupyterlab.labapp import LabApp
@pytest.fixture
def fetch_long(http_server_client, jp_auth_header, jp_base_url):
    """fetch fixture that handles auth, base_url, and path"""

    def client_fetch(*parts, headers=None, params=None, **kwargs):
        path_url = url_escape(url_path_join(*parts), plus=False)
        path_url = url_path_join(jp_base_url, path_url)
        params_url = urllib.parse.urlencode(params or {})
        url = path_url + '?' + params_url
        headers = headers or {}
        headers.update(jp_auth_header)
        return http_server_client.fetch(url, headers=headers, request_timeout=250, **kwargs)
    return client_fetch