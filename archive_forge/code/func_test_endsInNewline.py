from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_endsInNewline(self) -> None:
    """
        L{text.endsInNewline} returns C{True} if the string ends in a newline.
        """
    s = 'newline\n'
    m = text.endsInNewline(s)
    self.assertTrue(m)
    s = 'oldline'
    m = text.endsInNewline(s)
    self.assertFalse(m)