import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_base_url_new_version(self):
    version = 'v1'
    self.key_mgr.conf.barbican.barbican_api_version = version
    endpoint = 'http://localhost/key_manager'
    base_url = self.key_mgr._create_base_url(mock.Mock(), mock.Mock(), endpoint)
    self.assertEqual(endpoint + '/' + version, base_url)