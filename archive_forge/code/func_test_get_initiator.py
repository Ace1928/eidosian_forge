import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
def test_get_initiator(self):

    def initiator_no_file(*args, **kwargs):
        raise putils.ProcessExecutionError('No file')
    self.connector._execute = initiator_no_file
    initiator = self.connector.get_initiator()
    self.assertIsNone(initiator)
    self.connector._execute = self._initiator_get_text
    initiator = self.connector.get_initiator()
    self.assertEqual(initiator, self._fake_iqn)