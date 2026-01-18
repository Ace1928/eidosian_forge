from unittest import mock
import fixtures
from heat.common import exception
from heat.common.i18n import _
from heat.tests import common
def test_format_string_error_message(self):
    message = 'This format %(message)s should work'
    err = exception.Error(message)
    self.assertEqual(message, str(err))