from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_minus(self) -> None:
    """
        Formatting attributes prefixed with a minus (C{-}) temporarily disable
        the prefixed attribute, emitting no VT102 control sequence to enable
        it in the flattened output.
        """
    self.assertEqual(text.assembleFormattedText(A.bold[A.blink['Hello', -A.bold[' world'], '.']]), '\x1b[1;5mHello\x1b[0;5m world\x1b[1;5m.')