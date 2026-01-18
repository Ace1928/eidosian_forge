import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def ftp_FEAT(self):
    """
        Advertise the features supported by the server.

        http://tools.ietf.org/html/rfc2389
        """
    self.sendLine(RESPONSE[FEAT_OK][0])
    for feature in self.FEATURES:
        self.sendLine(' ' + feature)
    self.sendLine(RESPONSE[FEAT_OK][1])