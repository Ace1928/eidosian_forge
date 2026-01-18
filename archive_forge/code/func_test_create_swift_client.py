from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
@mock.patch.object(swiftclient.client, 'Connection')
def test_create_swift_client(self, sc_conn):
    auth = mock.MagicMock()
    auth.get_token.return_value = '1234'
    auth.get_endpoint.return_value = 'http://192.0.2.1:8080'
    session = mock.MagicMock()
    args = mock.MagicMock()
    args.os_region_name = 'Region1'
    args.os_project_name = 'project'
    args.os_username = 'user'
    args.os_cacert = None
    args.insecure = True
    sc_conn.return_value = mock.MagicMock()
    sc = deployment_utils.create_swift_client(auth, session, args)
    self.assertEqual(sc_conn.return_value, sc)
    self.assertEqual(mock.call(session), auth.get_token.call_args)
    self.assertEqual(mock.call(session, service_type='object-store', region_name='Region1'), auth.get_endpoint.call_args)
    self.assertEqual(mock.call(cacert=None, insecure=True, key=None, tenant_name='project', preauthtoken='1234', authurl=None, user='user', preauthurl='http://192.0.2.1:8080', auth_version='2.0'), sc_conn.call_args)