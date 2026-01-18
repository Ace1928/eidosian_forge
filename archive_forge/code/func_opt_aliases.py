import os
from twisted.application import internet
from twisted.cred import checkers, strcred
from twisted.internet import endpoints
from twisted.mail import alias, mail, maildir, relay, relaymanager
from twisted.python import usage
def opt_aliases(self, filename):
    """
        Specify an aliases(5) file to use for the last specified domain.
        """
    if self.last_domain is not None:
        if mail.IAliasableDomain.providedBy(self.last_domain):
            aliases = alias.loadAliasFile(self.service.domains, filename)
            self.last_domain.setAliasGroup(aliases)
            self.service.monitor.monitorFile(filename, AliasUpdater(self.service.domains, self.last_domain))
        else:
            raise usage.UsageError('%s does not support alias files' % (self.last_domain.__class__.__name__,))
    else:
        raise usage.UsageError('Specify a domain before specifying aliases')