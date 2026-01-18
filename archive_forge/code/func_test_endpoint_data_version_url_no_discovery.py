import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_endpoint_data_version_url_no_discovery(self):
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = a.get_endpoint_data(session=s, service_type='volumev3', interface='admin', discover_versions=False)
    self.assertEqual(self.TEST_VOLUME.versions['v3'].service.admin, data.url)
    self.assertEqual((3, 0), data.api_version)