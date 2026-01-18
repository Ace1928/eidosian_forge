from zope.interface import implementer
from twisted.copyright import longversion
from twisted.cred.credentials import CramMD5Credentials, UsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol
from twisted.mail import pop3, relay, smtp
from twisted.python import log
def lookupDomain(self, user):
    """
        Check whether a domain is among the virtual domains supported by the
        mail service.

        @type user: L{bytes}
        @param user: An email address.

        @rtype: 2-L{tuple} of (L{bytes}, L{bytes})
        @return: The local part and the domain part of the email address if the
            domain is supported.

        @raise POP3Error: When the domain is not supported by the mail service.
        """
    try:
        user, domain = user.split(self.domainSpecifier, 1)
    except ValueError:
        domain = b''
    if domain not in self.service.domains:
        raise pop3.POP3Error('no such domain {}'.format(domain.decode('utf-8')))
    return (user, domain)