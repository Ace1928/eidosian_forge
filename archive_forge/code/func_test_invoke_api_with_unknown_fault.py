from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_invoke_api_with_unknown_fault(self):
    api_session = self._create_api_session(True)
    fault_list = ['NotAFile']
    module = mock.Mock()
    module.api.side_effect = exceptions.VimFaultException(fault_list, 'Not a file.')
    ex = self.assertRaises(exceptions.VimFaultException, api_session.invoke_api, module, 'api')
    self.assertEqual(fault_list, ex.fault_list)