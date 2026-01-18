from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_custom_execute_check_exit_code(self):
    self.assertRaises(putils.ProcessExecutionError, priv_rootwrap.custom_execute, 'ls', '-y', check_exit_code=True)