from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(networkutils, 'patcher')
@mock.patch.object(networkutils.tpool, 'execute')
@mock.patch.object(networkutils.NetworkUtils, '_get_event_wql_query')
def test_get_vnic_event_listener(self, mock_get_event_query, mock_execute, mock_patcher):
    event = mock.MagicMock()
    unnamed_port_event = mock.MagicMock(ElementName=None)
    port_class = self.netutils._conn.Msvm_SyntheticEthernetPortSettingData
    wmi_event_listener = port_class.watch_for.return_value
    mock_execute.side_effect = [exceptions.x_wmi_timed_out, unnamed_port_event, event]
    callback = mock.MagicMock(side_effect=TypeError)
    returned_listener = self.netutils.get_vnic_event_listener(self.netutils.EVENT_TYPE_CREATE)
    self.assertRaises(TypeError, returned_listener, callback)
    mock_get_event_query.assert_called_once_with(cls=self.netutils._VNIC_SET_DATA, event_type=self.netutils.EVENT_TYPE_CREATE, timeframe=2)
    port_class.watch_for.assert_called_once_with(mock_get_event_query.return_value)
    mock_execute.assert_has_calls([mock.call(wmi_event_listener, self.netutils._VNIC_LISTENER_TIMEOUT_MS)] * 3)
    callback.assert_called_once_with(event.ElementName)