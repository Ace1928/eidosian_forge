import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_selection_spec(self):
    client_factory = mock.Mock()
    sel_spec = vim_util.build_selection_spec(client_factory, 'test')
    self.assertEqual('test', sel_spec.name)