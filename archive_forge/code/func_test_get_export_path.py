import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
def test_get_export_path(self):
    fake_export = '//ip/share'
    fake_conn_props = dict(export=fake_export)
    expected_export = fake_export.replace('/', '\\')
    export_path = self._connector._get_export_path(fake_conn_props)
    self.assertEqual(expected_export, export_path)