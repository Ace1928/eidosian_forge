import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_parse_string_list')
def test_get_initiators(self, mock_parse_string_list):
    self._mock_ctypes()
    get_initiators = _utils.get_wrapped_function(self._initiator.get_iscsi_initiators)
    mock_el_count = mock.Mock(value=mock.sentinel.element_count)
    resulted_initator_list = get_initiators(self._initiator, element_count=mock_el_count, buff=mock.sentinel.buff)
    self.assertEqual(mock_parse_string_list.return_value, resulted_initator_list)
    self._mock_run.assert_called_once_with(self._iscsidsc.ReportIScsiInitiatorListW, self._ctypes.byref(mock_el_count), mock.sentinel.buff)
    mock_parse_string_list.assert_called_once_with(mock.sentinel.buff, mock.sentinel.element_count)