from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_check_tag_not_exists(self):
    res = self.sot
    sess = self.session
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.links = {}
    mock_response.content = None
    sess.get.side_effect = [mock_response]
    self.assertRaises(exceptions.NotFoundException, res.check_tag, sess, 'dummy')