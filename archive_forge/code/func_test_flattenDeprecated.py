from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_flattenDeprecated(self) -> None:
    """
        L{twisted.conch.insults.text.flatten} emits a deprecation warning when
        imported or accessed.
        """
    warningsShown = self.flushWarnings([self.test_flattenDeprecated])
    self.assertEqual(len(warningsShown), 0)
    text.flatten
    warningsShown = self.flushWarnings([self.test_flattenDeprecated])
    self.assertEqual(len(warningsShown), 1)
    self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
    self.assertEqual(warningsShown[0]['message'], 'twisted.conch.insults.text.flatten was deprecated in Twisted 13.1.0: Use twisted.conch.insults.text.assembleFormattedText instead.')