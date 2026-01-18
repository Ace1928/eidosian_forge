from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(cloud_region.CloudRegion, 'get_session')
def test_get_session_endpoint_exception(self, get_session_mock):

    class FakeException(Exception):
        pass

    def side_effect(*args, **kwargs):
        raise FakeException('No service')
    session_mock = mock.Mock()
    session_mock.get_endpoint.side_effect = side_effect
    get_session_mock.return_value = session_mock
    self.cloud.name = 'testcloud'
    self.cloud.config.config['region_name'] = 'testregion'
    with testtools.ExpectedException(exceptions.SDKException, 'Error getting image endpoint on testcloud:testregion: No service'):
        self.cloud.get_session_endpoint('image')