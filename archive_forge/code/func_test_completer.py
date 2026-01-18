from twisted.python import usage
from twisted.trial import unittest
def test_completer(self):
    """
        Completer produces zsh shell-code that produces no completion matches.
        """
    c = usage.Completer()
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, ':some-option:')
    c = usage.Completer(descr='some action', repeat=True)
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, '*:some action:')