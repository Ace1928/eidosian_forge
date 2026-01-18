from twisted.python import reflect, win32
from twisted.trial import unittest
def test_argWithoutSpaces(self) -> None:
    """
        Calling C{cmdLineQuote} with an argument with no spaces returns
        the argument unchanged.
        """
    self.assertEqual(win32.cmdLineQuote('an_argument'), 'an_argument')