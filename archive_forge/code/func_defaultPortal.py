import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def defaultPortal(self):
    """
        Return the portal for the default domain.

        The default domain is named ''.

        @rtype: L{Portal}
        @return: The portal for the default domain.
        """
    return self.portals['']