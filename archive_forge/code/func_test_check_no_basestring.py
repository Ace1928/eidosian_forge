import testtools
from oslotest import base
from octavia_lib.hacking import checks
def test_check_no_basestring(self):
    self.assertEqual(1, len(list(checks.check_no_basestring("isinstance('foo', basestring)"))))
    self.assertEqual(0, len(list(checks.check_no_basestring("isinstance('foo', str)"))))