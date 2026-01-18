from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_erefused(self):
    """
        L{ResolverBase.exceptionForCode} converts L{EREFUSED} to
        L{DNSQueryRefusedError}.
        """
    self.assertIs(self.exceptionForCode(EREFUSED), DNSQueryRefusedError)