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
def test_basic_authentication(self):
    tuples = [('1.0', OpenStackMockHttp, {}), ('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'project', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'}), ('3.x_appcred', OpenStackIdentity_3_0_MockHttp, {'user_id': 'appcred_id', 'key': 'appcred_secret'}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'user_id': 'test_user_id', 'key': 'test_key', 'token_scope': 'domain', 'tenant_name': 'test_tenant', 'tenant_domain_id': 'test_tenant_domain_id', 'domain_name': 'test_domain'})]
    user_id = OPENSTACK_PARAMS[0]
    key = OPENSTACK_PARAMS[1]
    for auth_version, mock_http_class, kwargs in tuples:
        connection = self._get_mock_connection(mock_http_class=mock_http_class)
        auth_url = connection.auth_url
        if not kwargs:
            kwargs['user_id'] = user_id
            kwargs['key'] = key
        cls = get_class_for_auth_version(auth_version=auth_version)
        osa = cls(auth_url=auth_url, parent_conn=connection, **kwargs)
        self.assertEqual(osa.urls, {})
        self.assertIsNone(osa.auth_token)
        self.assertIsNone(osa.auth_user_info)
        osa = osa.authenticate()
        self.assertTrue(len(osa.urls) >= 1)
        self.assertTrue(osa.auth_token is not None)
        if auth_version in ['1.1', '2.0', '2.0_apikey', '2.0_password', '3.x_password', '3.x_appcred', '3.x_oidc_access_token']:
            self.assertTrue(osa.auth_token_expires is not None)
        if auth_version in ['2.0', '2.0_apikey', '2.0_password', '3.x_password', '3.x_appcred', '3.x_oidc_access_token']:
            self.assertTrue(osa.auth_user_info is not None)