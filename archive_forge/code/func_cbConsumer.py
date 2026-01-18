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
def cbConsumer(cons):
    """
            Called after the file was opended for reading.

            Prepare the data transfer channel and send the response
            to the command channel.
            """
    if not self.binary:
        cons = ASCIIConsumerWrapper(cons)
    d = self.dtpInstance.registerConsumer(cons)
    if self.dtpInstance.isConnected:
        self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
    else:
        self.reply(FILE_STATUS_OK_OPEN_DATA_CNX)
    return d