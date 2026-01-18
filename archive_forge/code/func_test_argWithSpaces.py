from twisted.python import reflect, win32
from twisted.trial import unittest
def test_argWithSpaces(self) -> None:
    """
        Calling C{cmdLineQuote} with an argument containing spaces returns
        the argument surrounded by quotes.
        """
    self.assertEqual(win32.cmdLineQuote('An Argument'), '"An Argument"')