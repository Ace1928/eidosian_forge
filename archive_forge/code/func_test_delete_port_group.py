import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_delete_port_group(self):
    session = mock.Mock()
    dvs_util.delete_port_group(session, 'pg-moref')
    session.invoke_api.assert_called_once_with(session.vim, 'Destroy_Task', 'pg-moref')