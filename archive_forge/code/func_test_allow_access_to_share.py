from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data({'access_to': '127.0.0.1', 'access_type': 'ip', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': '1' * 4, 'access_type': 'user', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': '1' * 255, 'access_type': 'user', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': "fake${.-_'`}", 'access_type': 'user', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': 'MYDOMAIN-Administrator', 'access_type': 'user', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': 'test group name', 'access_type': 'user', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': 'x', 'access_type': 'cert', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': 'x' * 64, 'access_type': 'cert', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': 'tenant.example.com', 'access_type': 'cert', 'action_name': 'os-allow_access', 'microversion': '2.0'}, {'access_to': '127.0.0.1', 'access_type': 'ip', 'action_name': 'allow_access', 'microversion': '2.7'}, {'access_to': 'test group name', 'access_type': 'user', 'action_name': 'allow_access', 'microversion': '2.7'}, {'access_to': 'alice', 'access_type': 'cephx', 'action_name': 'allow_access', 'microversion': '2.13'}, {'access_to': 'alice_bob', 'access_type': 'cephx', 'action_name': 'allow_access', 'microversion': '2.13'}, {'access_to': 'alice bob', 'access_type': 'cephx', 'action_name': 'allow_access', 'microversion': '2.13'}, {'access_to': 'test group name', 'access_type': 'user', 'action_name': 'allow_access', 'microversion': '2.13'}, {'access_to': 'AD80:0000:0000:0000:ABAA:0000:00C2:0002', 'access_type': 'ip', 'action_name': 'allow_access', 'microversion': '2.38'}, {'access_to': 'AD80::/36', 'access_type': 'ip', 'action_name': 'allow_access', 'microversion': '2.38'}, {'access_to': 'AD80:ABAA::/128', 'access_type': 'ip', 'action_name': 'allow_access', 'microversion': '2.38'}, {'access_to': 'ad80::abaa:0:c2:2', 'access_type': 'ip', 'action_name': 'allow_access', 'microversion': '2.38'}, {'access_to': 'test group name', 'access_type': 'user', 'action_name': 'allow_access', 'microversion': '2.38'})
@ddt.unpack
def test_allow_access_to_share(self, access_to, access_type, action_name, microversion):
    access = ('foo', {'access': 'bar'})
    access_level = 'fake_access_level'
    share = 'fake_share'
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value=access)):
        result = manager.allow(share, access_type, access_to, access_level)
        manager._action.assert_called_once_with(action_name, share, {'access_level': access_level, 'access_type': access_type, 'access_to': access_to})
        self.assertEqual('bar', result)