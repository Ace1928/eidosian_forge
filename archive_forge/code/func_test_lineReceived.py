import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_lineReceived(self):
    parser = details.SimpleDetailsParser(None)
    parser.lineReceived(_b('foo\n'))
    parser.lineReceived(_b('bar\n'))
    self.assertEqual(_b('foo\nbar\n'), parser._message)