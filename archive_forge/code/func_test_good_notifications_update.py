from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notifications
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_good_notifications_update(self, mc):
    mc.return_value = c = FakeV2Client()
    id_str = '0495340b-58fd-4e1c-932b-5e6f9cc96491'
    raw_args = '{0} notification_updated_name EMAIL john.doe@hpe.com 0'.format(id_str).split(' ')
    name, cmd_clazz = migr.create_command_class('do_notification_update', shell)
    cmd = cmd_clazz(mock.Mock(), mock.Mock())
    parser = cmd.get_parser(name)
    parsed_args = parser.parse_args(raw_args)
    cmd.run(parsed_args)
    data = {'name': 'notification_updated_name', 'type': 'EMAIL', 'address': 'john.doe@hpe.com', 'period': 0, 'notification_id': id_str}
    c.notifications.update.assert_called_once_with(**data)