from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_fetch_tags(self):
    res = self.sot
    sess = self.session
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.return_value = {'tags': ['blue1', 'green1']}
    sess.get.side_effect = [mock_response]
    result = res.fetch_tags(sess)
    self.assertEqual(['blue1', 'green1'], res.tags)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/tags'
    sess.get.assert_called_once_with(url)