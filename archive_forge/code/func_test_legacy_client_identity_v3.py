import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from os_client_config import cloud_config
from os_client_config import defaults
from os_client_config import exceptions
from os_client_config.tests import base
@mock.patch.object(cloud_region.CloudRegion, 'get_session_endpoint')
def test_legacy_client_identity_v3(self, mock_get_session_endpoint):
    mock_client = mock.Mock()
    mock_get_session_endpoint.return_value = 'http://example.com'
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    config_dict['identity_api_version'] = '3'
    cc = cloud_config.CloudConfig('test1', 'region-al', config_dict, auth_plugin=mock.Mock())
    cc.get_legacy_client('identity', mock_client)
    mock_client.assert_called_with(version='3', endpoint='http://example.com', interface='admin', endpoint_override=None, region_name='region-al', service_type='identity', session=mock.ANY, service_name='locks')