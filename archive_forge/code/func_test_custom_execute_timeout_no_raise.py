from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_custom_execute_timeout_no_raise(self):
    out, err = priv_rootwrap.custom_execute('sleep', '2', timeout=0.05, raise_timeout=False)
    self.assertEqual('', out)
    self.assertIsInstance(err, str)