from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data({'access_to': 'localhost', 'access_type': 'ip', 'microversion': '2.0'}, {'access_to': '127.0.0.*', 'access_type': 'ip', 'microversion': '2.0'}, {'access_to': '127.0.0.0/33', 'access_type': 'ip', 'microversion': '2.0'}, {'access_to': '127.0.0.256', 'access_type': 'ip', 'microversion': '2.0'}, {'access_to': '1', 'access_type': 'user', 'microversion': '2.0'}, {'access_to': '1' * 3, 'access_type': 'user', 'microversion': '2.0'}, {'access_to': '1' * 256, 'access_type': 'user', 'microversion': '2.0'}, {'access_to': 'root+=', 'access_type': 'user', 'microversion': '2.0'}, {'access_to': '', 'access_type': 'cert', 'microversion': '2.0'}, {'access_to': ' ', 'access_type': 'cert', 'microversion': '2.0'}, {'access_to': 'x' * 65, 'access_type': 'cert', 'microversion': '2.0'}, {'access_to': 'alice', 'access_type': 'cephx', 'microversion': '2.0'}, {'access_to': '', 'access_type': 'cephx', 'microversion': '2.13'}, {'access_to': u'björn', 'access_type': 'cephx', 'microversion': '2.13'}, {'access_to': 'AD80:0000:0000:0000:ABAA:0000:00C2:0002/65', 'access_type': 'ip', 'microversion': '2.38'}, {'access_to': 'AD80:0000:0000:0000:ABAA:0000:00C2:0002*32', 'access_type': 'ip', 'microversion': '2.38'}, {'access_to': 'AD80:0000:0000:0000:ABAA:0000:00C2:0002', 'access_type': 'ip', 'microversion': '2.37'})
@ddt.unpack
def test_allow_access_to_share_error_invalid_access(self, access_to, access_type, microversion):
    access = ('foo', {'access': 'bar'})
    access_level = 'fake_access_level'
    share = 'fake_share'
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value=access)):
        self.assertRaises(exceptions.CommandError, manager.allow, share, access_type, access_to, access_level)
        manager._action.assert_not_called()