from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_idmethod')
def test_already_associated_initiator(self, mock_get_wtidmethod):
    mock_wt_idmeth_cls = self._tgutils._conn_wmi.WT_IDMethod
    self._tgutils.associate_initiator_with_iscsi_target(mock.sentinel.initiator, mock.sentinel.target_name, id_method=mock.sentinel.id_method)
    self.assertFalse(mock_wt_idmeth_cls.new.called)