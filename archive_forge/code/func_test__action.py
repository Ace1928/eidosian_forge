from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
@ddt.data(('reset_status', {'status': constants.STATUS_AVAILABLE}), ('unmanage', {'id': 'fake_id'}))
@ddt.unpack
def test__action(self, action, info):
    action = ''
    share_server = {'id': 'fake_id'}
    expected_url = '/share-servers/%s/action' % share_server['id']
    expected_body = {action: info}
    with mock.patch.object(self.manager.api.client, 'post', mock.Mock(return_value='fake')):
        self.mock_object(base, 'getid', mock.Mock(return_value=share_server['id']))
        result = self.manager._action(action, share_server, info)
        self.manager.api.client.post.assert_called_once_with(expected_url, body=expected_body)
        self.assertEqual('fake', result)