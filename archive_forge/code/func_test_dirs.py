from twisted.python import usage
from twisted.trial import unittest
def test_dirs(self):
    """
        CompleteDirs produces zsh shell-code that completes directory names.
        """
    c = usage.CompleteDirs()
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, ':some-option:_directories')
    c = usage.CompleteDirs(descr='some action', repeat=True)
    got = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(got, '*:some action:_directories')