from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_exception_summary_exception_as_list(self):
    self.assertRaises(ValueError, exceptions.VimException, [], ValueError('foo'))