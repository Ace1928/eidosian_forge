import os
import warnings
from zope.interface import implementer
from twisted.application import internet, service
from twisted.cred.portal import Portal
from twisted.internet import defer
from twisted.mail import protocols, smtp
from twisted.mail.interfaces import IAliasableDomain, IDomain
from twisted.python import log, util
def setDefaultDomain(self, domain):
    """
        Set the default domain.

        @type domain: L{IDomain} provider
        @param domain: The default domain.
        """
    self.default = domain