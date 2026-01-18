from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_get_metadata_item(self):
    res = self.sot
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'meta': {'foo': 'bar'}}
    self.session.get.side_effect = [mock_response]
    result = res.get_metadata_item(self.session, 'foo')
    self.assertEqual({'foo': 'bar'}, res.metadata)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/metadata/foo'
    self.session.get.assert_called_once_with(url)