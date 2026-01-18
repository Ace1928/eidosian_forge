from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@ddt.data({'auth_url': 'only_v3', 'password': 'password_backward_compat', 'endpoint_type': 'publicURL', 'project_id': 'foo_tenant_project_id'}, {'password': 'renamed_api_key', 'endpoint_type': 'public', 'tenant_id': 'foo_tenant_project_id'})
def test_client_init_no_session_no_auth_token_v3(self, kwargs):

    def fake_url_for(version):
        if version == 'v3.0':
            return 'url_v3.0'
        elif version == 'v2.0' and self.auth_url == 'both':
            return 'url_v2.0'
        else:
            return None
    self.mock_object(client.httpclient, 'HTTPClient')
    self.mock_object(client.ks_client, 'Client')
    self.mock_object(client.session.discover, 'Discover')
    self.mock_object(client.session, 'Session')
    client_args = self._get_client_args(**kwargs)
    client_args['api_version'] = manilaclient.API_MIN_VERSION
    self.auth_url = client_args['auth_url']
    catalog = {'share': [{'region': 'SecondRegion', 'region_id': 'SecondRegion', 'url': 'http://4.4.4.4', 'interface': 'public'}], 'sharev2': [{'region': 'FirstRegion', 'interface': 'public', 'region_id': 'SecondRegion', 'url': 'http://1.1.1.1'}, {'region': 'secondregion', 'interface': 'public', 'region_id': 'SecondRegion', 'url': 'http://2.2.2.2'}, {'region': 'SecondRegion', 'interface': 'internal', 'region_id': 'SecondRegion', 'url': 'http://3.3.3.1'}, {'region': 'SecondRegion', 'interface': 'public', 'region_id': 'SecondRegion', 'url': 'http://3.3.3.3'}, {'region': 'SecondRegion', 'interface': 'admin', 'region_id': 'SecondRegion', 'url': 'http://3.3.3.2'}]}
    client.session.discover.Discover.return_value.url_for.side_effect = fake_url_for
    client.ks_client.Client.return_value.auth_token.return_value = 'fake_token'
    mocked_ks_client = client.ks_client.Client.return_value
    mocked_ks_client.service_catalog.get_endpoints.return_value = catalog
    client.Client(**client_args)
    client.httpclient.HTTPClient.assert_called_with('http://3.3.3.3', mock.ANY, 'python-manilaclient', insecure=False, cacert=None, cert=client_args['cert'], timeout=None, retries=None, http_log_debug=False, api_version=manilaclient.API_MIN_VERSION)
    client.ks_client.Client.assert_called_with(session=mock.ANY, version=(3, 0), auth_url='url_v3.0', username=client_args['username'], password=client_args.get('password'), user_id=client_args['user_id'], user_domain_name=client_args['user_domain_name'], user_domain_id=client_args['user_domain_id'], project_id=client_args.get('tenant_id', client_args.get('project_id')), project_name=client_args['project_name'], project_domain_name=client_args['project_domain_name'], project_domain_id=client_args['project_domain_id'], region_name=client_args['region_name'])
    mocked_ks_client.service_catalog.get_endpoints.assert_called_with(client_args['service_type'])
    mocked_ks_client.authenticate.assert_called_with()