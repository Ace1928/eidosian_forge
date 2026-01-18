import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_create_if_missing(self):
    ignore_path = bedding.user_ignore_config_path()
    self.assertPathDoesNotExist(ignore_path)
    user_ignores = ignores.get_user_ignores()
    self.assertEqual(set(ignores.USER_DEFAULTS), user_ignores)
    self.assertPathExists(ignore_path)
    with open(ignore_path, 'rb') as f:
        entries = ignores.parse_ignore_file(f)
    self.assertEqual(set(ignores.USER_DEFAULTS), entries)