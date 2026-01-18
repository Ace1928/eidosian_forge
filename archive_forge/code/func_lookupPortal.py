import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def lookupPortal(self, name):
    """
        Find the portal for a domain.

        @type name: L{bytes}
        @param name: A domain name.

        @rtype: L{Portal}
        @return: A portal.
        """
    return self.portals[name]