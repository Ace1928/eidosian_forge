from twisted.conch.insults import text
from twisted.conch.insults.text import attributes as A
from twisted.trial import unittest
def test_background(self) -> None:
    """
        The background color formatting attribute, L{A.bg}, emits the VT102
        control sequence to set the selected background color when flattened.
        """
    self.assertEqual(text.assembleFormattedText(A.normal[A.bg.red['Hello, '], A.bg.green['world!']]), '\x1b[41mHello, \x1b[42mworld!')