from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@ddt.data({'auth_url': 'only_v2', 'password': 'foo', 'project_id': 'bar'}, {'password': 'foo', 'tenant_id': 'bar'})
def test_client_init_no_session_no_auth_token_v2(self, kwargs):
    self.mock_object(client.httpclient, 'HTTPClient')
    self.mock_object(client.ks_client, 'Client')
    self.mock_object(client.session.discover, 'Discover')
    self.mock_object(client.session, 'Session')
    client_args = self._get_client_args(**kwargs)
    client_args['api_version'] = manilaclient.API_MIN_VERSION
    self.auth_url = client_args['auth_url']
    catalog = {'share': [{'region': 'SecondRegion', 'publicUrl': 'http://4.4.4.4'}], 'sharev2': [{'region': 'FirstRegion', 'publicUrl': 'http://1.1.1.1'}, {'region': 'secondregion', 'publicUrl': 'http://2.2.2.2'}, {'region': 'SecondRegion', 'internalUrl': 'http://3.3.3.1', 'publicUrl': 'http://3.3.3.3', 'adminUrl': 'http://3.3.3.2'}]}
    client.session.discover.Discover.return_value.url_for.side_effect = lambda v: 'url_v2.0' if v == 'v2.0' else None
    client.ks_client.Client.return_value.auth_token.return_value = 'fake_token'
    mocked_ks_client = client.ks_client.Client.return_value
    mocked_ks_client.service_catalog.get_endpoints.return_value = catalog
    client.Client(**client_args)
    client.httpclient.HTTPClient.assert_called_with('http://3.3.3.3', mock.ANY, 'python-manilaclient', insecure=False, cacert=None, cert=client_args['cert'], timeout=None, retries=None, http_log_debug=False, api_version=manilaclient.API_MIN_VERSION)
    client.ks_client.Client.assert_called_with(session=mock.ANY, version=(2, 0), auth_url='url_v2.0', username=client_args['username'], password=client_args.get('password'), tenant_id=client_args.get('tenant_id', client_args.get('project_id')), tenant_name=client_args['project_name'], region_name=client_args['region_name'], cert=client_args['cert'], use_keyring=False, force_new_token=False, stale_duration=300)
    mocked_ks_client.service_catalog.get_endpoints.assert_called_with(client_args['service_type'])
    mocked_ks_client.authenticate.assert_called_with()