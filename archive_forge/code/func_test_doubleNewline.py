from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_doubleNewline(self) -> None:
    """
        Allow paragraphs delimited by two 
s.
        """
    sampleText = 'et\n\nphone\nhome.'
    result = text.wordWrap(sampleText, self.lineWidth)
    self.assertEqual(result, ['et', '', 'phone home.', ''])