from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_multiWord(self) -> None:
    s = 'The "hairy monkey" likes pie.'
    r = text.splitQuoted(s)
    self.assertEqual(['The', 'hairy monkey', 'likes', 'pie.'], r)