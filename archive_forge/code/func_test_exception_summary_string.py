from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_exception_summary_string(self):
    e = exceptions.VimException(_('string'), ValueError('foo'))
    string = str(e)
    self.assertEqual('string\nCause: foo', string)