import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_init_failure(self):
    session = self._create_mock_session(False)
    self.assertRaises(exceptions.VimException, rw_handles.VmdkReadHandle, session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', 100)