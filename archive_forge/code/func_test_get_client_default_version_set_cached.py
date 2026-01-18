from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_default_version_set_cached(self):
    version = '1.3'
    self.assertNotEqual(v1.DEFAULT_VER, version)
    kwargs = {'project_name': 'PROJECT_NAME', 'username': 'USERNAME', 'password': 'PASSWORD', 'auth_url': 'http://localhost:35357/v2.0'}
    self._test_get_client(version=version, **kwargs)