import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_bounce_to_postmaster(self):
    """
        Send undeliverable messages to the postmaster.
        """
    self.last_domain.postmaster = 1