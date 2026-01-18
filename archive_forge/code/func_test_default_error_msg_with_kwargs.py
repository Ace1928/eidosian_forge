import http.client as http
from oslo_utils import encodeutils
from glance.common import exception
from glance.tests import utils as test_utils
def test_default_error_msg_with_kwargs(self):

    class FakeGlanceException(exception.GlanceException):
        message = 'default message: %(code)s'
    exc = FakeGlanceException(code=int(http.INTERNAL_SERVER_ERROR))
    self.assertEqual('default message: 500', encodeutils.exception_to_unicode(exc))