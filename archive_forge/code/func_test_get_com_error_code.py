from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils, 'get_com_error_hresult')
@mock.patch.object(_utils, 'hresult_to_err_code')
def test_get_com_error_code(self, mock_hres_to_err_code, mock_get_hresult):
    ret_val = _utils.get_com_error_code(mock.sentinel.com_err)
    self.assertEqual(mock_hres_to_err_code.return_value, ret_val)
    mock_get_hresult.assert_called_once_with(mock.sentinel.com_err)
    mock_hres_to_err_code.assert_called_once_with(mock_get_hresult.return_value)