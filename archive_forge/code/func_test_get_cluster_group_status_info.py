import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_get_cluster_group_status_info(self):
    prop_list = self._get_fake_prop_list()
    status_info = self._clusapi_utils.get_cluster_group_status_info(ctypes.byref(prop_list), ctypes.sizeof(prop_list))
    self.assertEqual(w_const.CLUSGRP_STATUS_WAITING_IN_QUEUE_FOR_MOVE, status_info)