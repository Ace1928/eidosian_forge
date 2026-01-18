from unittest import mock
from mistralclient.api.v2 import cron_triggers
from mistralclient.commands.v2 import cron_triggers as cron_triggers_cmd
from mistralclient.tests.unit import base
@mock.patch('mistralclient.commands.v2.cron_triggers.time')
def test_convert_time_string_to_utc_no_dst(self, mock_time):
    cmd = cron_triggers_cmd.Create(self.app, None)
    mock_time.daylight = 1
    mock_time.altzone = 4 * 60 * 60
    mock_time.timezone = 5 * 60 * 60
    mock_localtime = mock.Mock()
    mock_localtime.tm_isdst = 0
    mock_time.localtime.return_value = mock_localtime
    utc_value = cmd._convert_time_string_to_utc('4242-12-20 13:37')
    expected_time = '4242-12-20 18:37'
    self.assertEqual(expected_time, utc_value)