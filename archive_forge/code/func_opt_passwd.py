import socket
import sys
from typing import List, Optional, Sequence
from twisted import plugin
from twisted.application import strports
from twisted.application.service import MultiService
from twisted.cred import checkers, credentials, portal, strcred
from twisted.python import usage
from twisted.words import iwords, service
def opt_passwd(self, filename):
    """
        Name of a passwd-style file. (This is for
        backwards-compatibility only; you should use the --auth
        command instead.)
        """
    self.addChecker(checkers.FilePasswordDB(filename))