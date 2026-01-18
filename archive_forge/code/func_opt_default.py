import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_default(self):
    """
        Make the most recently specified domain the default domain.
        """
    if self.last_domain:
        self.service.addDomain('', self.last_domain)
    else:
        raise usage.UsageError('Specify a domain before specifying using --default')