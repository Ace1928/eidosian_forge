import os
import sys
from io import StringIO
from unittest import skipIf
from twisted.copyright import version
from twisted.internet.defer import Deferred
from twisted.internet.testing import MemoryReactor
from twisted.mail import smtp
from twisted.mail.scripts import mailmail
from twisted.mail.scripts.mailmail import parseOptions
from twisted.python.failure import Failure
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
def getConfigFromFile(self, config):
    """
        Read a mailmail configuration file.

        The mailmail script checks the twisted.mail.scripts.mailmail.GLOBAL_CFG
        variable and then the twisted.mail.scripts.mailmail.LOCAL_CFG
        variable for the path to its  config file.

        @param config: path to config file
        @type config: L{str}

        @return: A parsed config.
        @rtype: L{twisted.mail.scripts.mailmail.Configuration}
        """
    from twisted.mail.scripts.mailmail import loadConfig
    filename = self.mktemp()
    with open(filename, 'w') as f:
        f.write(config)
    return loadConfig(filename)