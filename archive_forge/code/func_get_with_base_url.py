from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_with_base_url(self, url, **kw):
    if 'default-types' in url:
        return self._cs_request(url, 'GET', **kw)
    server_versions = _stub_server_versions()
    return (200, {'versions': server_versions})