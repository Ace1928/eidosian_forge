import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from os_client_config import cloud_config
from os_client_config import defaults
from os_client_config import exceptions
from os_client_config.tests import base
@mock.patch.object(cloud_region.CloudRegion, 'get_auth_args')
@mock.patch.object(cloud_region.CloudRegion, 'get_session_endpoint')
def test_legacy_client_object_store_timeout(self, mock_get_session_endpoint, mock_get_auth_args):
    mock_client = mock.Mock()
    mock_get_session_endpoint.return_value = 'http://example.com/v2'
    mock_get_auth_args.return_value = {}
    config_dict = defaults.get_defaults()
    config_dict.update(fake_services_dict)
    config_dict['api_timeout'] = 9
    cc = cloud_config.CloudConfig(name='test1', region_name='region-al', config=config_dict, auth_plugin=mock.Mock())
    cc.get_legacy_client('object-store', mock_client)
    mock_client.assert_called_with(session=mock.ANY, os_options={'region_name': 'region-al', 'service_type': 'object-store', 'object_storage_url': None, 'endpoint_type': 'public'})