import os.path
import subprocess
import sys
from unittest import mock
from oslo_config import cfg
from oslotest import base
from oslo_upgradecheck import upgradecheck
def test_main_exception(self):
    raises = mock.Mock()
    raises.check.side_effect = Exception('test exception')
    self._run_test(raises, 255)