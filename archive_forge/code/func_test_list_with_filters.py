import testtools
from unittest import mock
from aodhclient.v2 import alarm
@mock.patch.object(alarm.AlarmManager, '_get')
def test_list_with_filters(self, mock_am):
    am = alarm.AlarmManager(self.client)
    filters = dict(type='gnocchi_resources_threshold', severity='low')
    am.list(filters=filters)
    expected_url = 'v2/alarms?q.field=severity&q.op=eq&q.value=low&q.field=type&q.op=eq&q.value=gnocchi_resources_threshold'
    mock_am.assert_called_with(expected_url)