import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
def test_is_guarded_host_config_error(self):
    self._test_is_host_guarded(return_code=mock.sentinel.return_code)