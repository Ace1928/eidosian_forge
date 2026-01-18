import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_smtp(self, description):
    """
        Add an SMTP port listener on the specified endpoint.

        You can listen on multiple ports by specifying multiple --smtp options.
        """
    self.addEndpoint('smtp', description)