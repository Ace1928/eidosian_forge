from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(cloud_region.CloudRegion, 'get_session')
def test_get_session_endpoint_unavailable(self, get_session_mock):
    session_mock = mock.Mock()
    session_mock.get_endpoint.return_value = None
    get_session_mock.return_value = session_mock
    image_endpoint = self.cloud.get_session_endpoint('image')
    self.assertIsNone(image_endpoint)