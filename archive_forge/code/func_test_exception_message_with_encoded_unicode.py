import sys
from unittest import mock
from oslo_utils import encodeutils
import testtools
from neutronclient._i18n import _
from neutronclient.common import exceptions
def test_exception_message_with_encoded_unicode(self):

    class TestException(exceptions.NeutronException):
        message = _('Exception with %(reason)s')
    multibyte_string = u'ＡＢＣ'
    multibyte_binary = encodeutils.safe_encode(multibyte_string)
    e = TestException(reason=multibyte_binary)
    self.assertEqual('Exception with %s' % multibyte_string, str(e))