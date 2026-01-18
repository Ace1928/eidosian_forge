from unittest import mock
from mistralclient.api.v2 import cron_triggers
from mistralclient.commands.v2 import cron_triggers as cron_triggers_cmd
from mistralclient.tests.unit import base
@mock.patch('mistralclient.commands.v2.cron_triggers.Create._convert_time_string_to_utc')
@mock.patch('argparse.open', create=True)
def test_create_no_utc(self, mock_open, mock_convert):
    self.client.cron_triggers.create.return_value = TRIGGER
    mock_open.return_value = mock.MagicMock(spec=open)
    mock_convert.return_value = '4242-12-20 18:37'
    result = self.call(cron_triggers_cmd.Create, app_args=['my_trigger', 'flow1', '--pattern', '* * * * *', '--params', '{}', '--count', '5', '--first-time', '4242-12-20 13:37'])
    mock_convert.assert_called_once_with('4242-12-20 13:37')
    self.client.cron_triggers.create.assert_called_once_with('my_trigger', 'flow1', {}, {}, '* * * * *', '4242-12-20 18:37', 5)
    self.assertEqual(('my_trigger', 'flow1', {}, '* * * * *', '4242-12-20 13:37', 5, '1', '1'), result[1])