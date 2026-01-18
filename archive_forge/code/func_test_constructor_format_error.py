from unittest import mock
from oslo_versionedobjects import exception
from oslo_versionedobjects import test
def test_constructor_format_error(self):
    with mock.patch.object(exception, 'LOG') as log:
        exc = exception.ObjectActionError()
        log.error.assert_called_with('code: 500')
    self.assertEqual(exception.ObjectActionError.msg_fmt, str(exc))