from __future__ import annotations
from twisted.internet.abstract import isIPv6Address
from twisted.trial.unittest import SynchronousTestCase
def test_invalidWithScopeID(self) -> None:
    """
        An otherwise invalid IPv6 address literal is still invalid with a
        trailing scope identifier.
        """
    self.assertFalse(isIPv6Address('%eth0'))
    self.assertFalse(isIPv6Address(':%eth0'))
    self.assertFalse(isIPv6Address('hello%eth0'))