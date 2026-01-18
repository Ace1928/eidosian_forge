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
def test_get_login_opts(self):
    fake_username = 'fake_chap_username'
    fake_password = 'fake_chap_secret'
    auth_type = constants.ISCSI_CHAP_AUTH_TYPE
    login_flags = w_const.ISCSI_LOGIN_FLAG_MULTIPATH_ENABLED
    login_opts = self._initiator._get_login_opts(auth_username=fake_username, auth_password=fake_password, auth_type=auth_type, login_flags=login_flags)
    self.assertEqual(len(fake_username), login_opts.UsernameLength)
    self.assertEqual(len(fake_password), login_opts.PasswordLength)
    username_struct_contents = ctypes.cast(login_opts.Username, ctypes.POINTER(ctypes.c_char * len(fake_username))).contents.value
    pwd_struct_contents = ctypes.cast(login_opts.Password, ctypes.POINTER(ctypes.c_char * len(fake_password))).contents.value
    self.assertEqual(six.b(fake_username), username_struct_contents)
    self.assertEqual(six.b(fake_password), pwd_struct_contents)
    expected_info_bitmap = w_const.ISCSI_LOGIN_OPTIONS_USERNAME | w_const.ISCSI_LOGIN_OPTIONS_PASSWORD | w_const.ISCSI_LOGIN_OPTIONS_AUTH_TYPE
    self.assertEqual(expected_info_bitmap, login_opts.InformationSpecified)
    self.assertEqual(login_flags, login_opts.LoginFlags)