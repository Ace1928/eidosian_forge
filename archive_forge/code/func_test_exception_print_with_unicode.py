import sys
from unittest import mock
from oslo_utils import encodeutils
import testtools
from neutronclient._i18n import _
from neutronclient.common import exceptions
def test_exception_print_with_unicode(self):

    class TestException(exceptions.NeutronException):
        message = _('Exception with %(reason)s')
    multibyte_unicode_string = u'ＡＢＣ'
    e = TestException(reason=multibyte_unicode_string)
    with mock.patch.object(sys, 'stdout') as mock_stdout:
        print(e)
    exc_str = 'Exception with %s' % multibyte_unicode_string
    mock_stdout.assert_has_calls([mock.call.write(exc_str)])