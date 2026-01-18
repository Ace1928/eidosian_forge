from twisted.python import usage
from twisted.trial import unittest
def test_hostnames(self):
    """
        CompleteHostnames produces zsh shell-code that completes hostnames.
        """
    c = usage.CompleteHostnames()
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, ':some-option:_hosts')
    c = usage.CompleteHostnames(descr='some action', repeat=True)
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, '*:some action:_hosts')