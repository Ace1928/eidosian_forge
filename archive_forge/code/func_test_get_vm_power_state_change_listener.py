from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_vm_power_state_change_listener(self):
    with mock.patch.object(self._vmutils, '_get_event_wql_query') as mock_get_query:
        listener = self._vmutils.get_vm_power_state_change_listener(timeframe=mock.sentinel.timeframe, filtered_states=mock.sentinel.filtered_states)
        mock_get_query.assert_called_once_with(cls=self._vmutils._COMPUTER_SYSTEM_CLASS, field=self._vmutils._VM_ENABLED_STATE_PROP, timeframe=mock.sentinel.timeframe, filtered_states=mock.sentinel.filtered_states)
        watcher = self._vmutils._conn.Msvm_ComputerSystem.watch_for
        watcher.assert_called_once_with(raw_wql=mock_get_query.return_value, fields=[self._vmutils._VM_ENABLED_STATE_PROP])
        self.assertEqual(watcher.return_value, listener)