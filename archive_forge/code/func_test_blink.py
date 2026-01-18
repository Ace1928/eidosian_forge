from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_blink(self) -> None:
    """
        The blink formatting attribute, L{A.blink}, emits the VT102 control
        sequence to enable blinking when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.blink['Hello, world.']), '\x1b[5mHello, world.')