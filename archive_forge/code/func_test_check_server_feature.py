from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_check_server_feature(self):
    mock_sv_feature_cls = self._hostutils._conn_cimv2.Win32_ServerFeature
    mock_sv_feature_cls.return_value = [mock.sentinel.sv_feature]
    feature_enabled = self._hostutils.check_server_feature(mock.sentinel.feature_id)
    self.assertTrue(feature_enabled)
    mock_sv_feature_cls.assert_called_once_with(ID=mock.sentinel.feature_id)