import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_close_with_error(self):
    session = self._create_mock_session()
    handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', 100)
    session.invoke_api.side_effect = exceptions.VimException(None)
    self.assertRaises(exceptions.VimException, handle.close)
    self._resp.close.assert_called_once_with()