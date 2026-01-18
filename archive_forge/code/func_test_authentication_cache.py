import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def test_authentication_cache(self):
    tuples = [('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'project', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'domain', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'})]
    user_id = OPENSTACK_PARAMS[0]
    key = OPENSTACK_PARAMS[1]
    for auth_version, mock_http_class, kwargs in tuples:
        mock_http_class.type = None
        connection = self._get_mock_connection(mock_http_class=mock_http_class)
        auth_url = connection.auth_url
        if not kwargs:
            kwargs['user_id'] = user_id
            kwargs['key'] = key
        auth_cache = OpenStackMockAuthCache()
        self.assertEqual(len(auth_cache), 0)
        kwargs['auth_cache'] = auth_cache
        cls = get_class_for_auth_version(auth_version=auth_version)
        osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
        osa = osa.authenticate()
        self.assertEqual(len(auth_cache), 1)
        osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
        osa.request = Mock(wraps=osa.request)
        osa = osa.authenticate()
        if auth_version in ('1.1', '2.0', '2.0_apikey', '2.0_password'):
            self.assertEqual(osa.request.call_count, 0)
        elif auth_version in ('3.x_password', '3.x_oidc_access_token'):
            osa.request.assert_called_once_with(action='/v3/auth/tokens', params=None, data=None, headers={'X-Subject-Token': '00000000000000000000000000000000', 'X-Auth-Token': '00000000000000000000000000000000'}, method='GET', raw=False)
        self.assertEqual(len(auth_cache), 1)
        cache_key = list(auth_cache.store.keys())[0]
        auth_context = auth_cache.get(cache_key)
        auth_context.expiration = YESTERDAY
        auth_cache.put(cache_key, auth_context)
        osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
        osa.request = Mock(wraps=osa.request)
        osa._get_unscoped_token_from_oidc_token = Mock(return_value='000')
        OpenStackIdentity_3_0_MockHttp.type = 'GET_UNAUTHORIZED_POST_OK'
        osa = osa.authenticate()
        if auth_version in ('1.1', '2.0', '2.0_apikey', '2.0_password'):
            self.assertEqual(osa.request.call_count, 1)
            self.assertTrue(osa.request.call_args[1]['method'], 'POST')
        elif auth_version in ('3.x_password', '3.x_oidc_access_token'):
            self.assertTrue(osa.request.call_args[0][0], '/v3/auth/tokens')
            self.assertTrue(osa.request.call_args[1]['method'], 'POST')
        if hasattr(osa, 'list_projects'):
            mock_http_class.type = None
            auth_cache.reset()
            osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
            osa.request = Mock(wraps=osa.request)
            osa = osa.authenticate()
            self.assertEqual(len(auth_cache), 1)
            mock_http_class.type = 'UNAUTHORIZED'
            try:
                osa.list_projects()
            except:
                pass
            self.assertEqual(len(auth_cache), 0)