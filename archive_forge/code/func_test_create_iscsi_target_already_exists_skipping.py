from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_create_iscsi_target_already_exists_skipping(self):
    self._test_create_iscsi_target_exception(target_exists=True)