from unittest import mock
from novaclient import api_versions
from novaclient import exceptions as exc
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import versions
def test_list_versions(self):
    fapi = mock.Mock()
    version_mgr = versions.VersionManager(fapi)
    version_mgr._list = mock.Mock()
    data = [('https://example.com:777/v2', 'https://example.com:777'), ('https://example.com/v2', 'https://example.com'), ('http://example.com/compute/v2', 'http://example.com/compute'), ('https://example.com/v2/prrrooojeect-uuid', 'https://example.com'), ('https://example.com:777/v2.1', 'https://example.com:777'), ('https://example.com/v2.1', 'https://example.com'), ('http://example.com/compute/v2.1', 'http://example.com/compute'), ('https://example.com/v2.1/prrrooojeect-uuid', 'https://example.com'), ('http://example.com/compute', 'http://example.com/compute'), ('http://compute.example.com', 'http://compute.example.com')]
    for endpoint, expected in data:
        version_mgr._list.reset_mock()
        fapi.client.get_endpoint.return_value = endpoint
        version_mgr.list()
        version_mgr._list.assert_called_once_with(expected, 'versions')