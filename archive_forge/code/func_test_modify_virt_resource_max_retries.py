from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_modify_virt_resource_max_retries(self):
    side_effect = [exceptions.HyperVException('expected failure.')] * 5 + [(self._FAKE_JOB_PATH, mock.MagicMock(), self._FAKE_RET_VAL)]
    self._check_modify_virt_resource_max_retries(side_effect=side_effect, num_calls=5)