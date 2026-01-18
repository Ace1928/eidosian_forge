from twisted.python import usage
from twisted.trial import unittest
def test_netInterfaces(self):
    """
        CompleteNetInterfaces produces zsh shell-code that completes system
        network interface names.
        """
    c = usage.CompleteNetInterfaces()
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, ':some-option:_net_interfaces')
    c = usage.CompleteNetInterfaces(descr='some action', repeat=True)
    out = c._shellCode('some-option', usage._ZSH)
    self.assertEqual(out, '*:some action:_net_interfaces')